"""
pipeline.py
-----------
End-to-end orchestrator for ExplainDL (MVP).
Exposes `run_pipeline` which:
 - accepts an uploaded file (path string or file-like object),
 - detects data type (tabular / image / text),
 - loads and preprocesses data,
 - selects and trains a suitable model,
 - generates explainability visualizations and a PDF report,
 - returns a results dictionary consumable by the Streamlit front-end.

This is designed as a pragmatic, extendable MVP — refine each step as you add features.
"""

import os
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd

from explainDL.data_input import detect_type, tabular_loader, image_loader, text_loader
from explainDL.preprocessing import tabular_preprocess, image_preprocess, text_preprocess
from explainDL.model_selection import auto_model_selector, tuner
from explainDL.training.trainer import train_and_evaluate
from explainDL.explainability import shap_explainer, lime_explainer, gradcam_explainer, report_generator
from explainDL.utils.file_utils import ensure_dir

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def _save_uploaded_file(uploaded_file, target_dir):
    """
    Save a Streamlit UploadedFile or file-like object to disk and return path.
    If uploaded_file is already a path string, return it unchanged.
    """
    if isinstance(uploaded_file, str):
        return uploaded_file

    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, uploaded_file.name)
    # UploadedFile-like: has .read()
    # reset pointer if possible
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    with open(out_path, "wb") as f:
        f.write(uploaded_file.read())
    return out_path


def _parse_text_lines_to_xy(lines):
    """
    Parse lines to (texts, labels). Accepts lines where each line is:
      <label> <sep> <text>
    Supported separators: tab, comma. Falls back to raise if no labels found.
    """
    texts = []
    labels = []
    for line in lines:
        if "\t" in line:
            lbl, txt = line.split("\t", 1)
        elif "," in line:
            # assume first token is label if there are at least 2 tokens
            parts = line.split(",", 1)
            if len(parts) == 2:
                lbl, txt = parts
            else:
                continue
        else:
            continue
        lbl = lbl.strip()
        txt = txt.strip()
        if lbl == "" or txt == "":
            continue
        texts.append(txt)
        labels.append(lbl)
    if len(texts) == 0:
        raise ValueError("Could not parse labels from text file. Expected lines with 'label<tab>text' or 'label, text'.")
    return texts, labels


def _save_matplotlib_figure(fig, path):
    """
    Save a matplotlib figure to path and close it.
    """
    fig.savefig(path, bbox_inches="tight")
    try:
        # a polite close to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def run_pipeline(uploaded_file,
                 auto_mode: bool = True,
                 show_explainability: bool = True,
                 enable_tuning: bool = False,
                 max_trials: int = 10):
    """
    Main entrypoint for ExplainDL pipeline.

    Parameters
    ----------
    uploaded_file : str or file-like
        Path to a file or file-like object (Streamlit UploadedFile)
    auto_mode : bool
        If True, use automatic heuristics for preprocessing & selection
    show_explainability : bool
        If True, attempt to compute explainability visualizations
    enable_tuning : bool
        If True, run the basic tuner where implemented
    max_trials : int
        Tuning budget (if tuning enabled)

    Returns
    -------
    dict
        {
            "metrics": pandas.DataFrame,
            "explainability_fig": <path to png>,
            "report_path": <path to pdf>,
            "model_name": str,
            "raw_metrics": dict
        }
    """

    temp_dir = tempfile.mkdtemp(prefix="explaindl_")
    ensure_dir(temp_dir)
    results = {}

    try:
        saved_path = _save_uploaded_file(uploaded_file, temp_dir)

        # detect type
        data_type = detect_type.detect_dataset_type(saved_path)
        print(f"[ExplainDL] Detected dataset type: {data_type}")

        # load & preprocess
        model = None
        model_name = None
        trained = None
        explain_fig_path = None
        report_path = None

        # initialize explainability objects (figs) to None so they're always defined
        shap_fig = None
        lime_fig = None
        grad_fig = None

        if data_type == "tabular":
            # load
            df = tabular_loader.load_tabular_data(saved_path)

            # preprocess (auto selects last column as target if not provided)
            X_train, X_test, y_train, y_test = tabular_preprocess.preprocess_tabular_data(df)

            # input shape for tabular MLP
            input_shape = (X_train.shape[1],)

            # ---------------------------
            # Robust label handling block
            # ---------------------------
            # Ensure numpy arrays for labels
            unique_labels = np.unique(y_train)
            num_classes = len(unique_labels)
            label_map = {v: i for i, v in enumerate(unique_labels)}

            # Convert to integer indices
            y_train_idx = np.array([label_map[v] for v in y_train])
            y_test_idx = np.array([label_map[v] for v in y_test])

            # One-hot encode only for multi-class (>2)
            if num_classes > 2:
                y_train_final = to_categorical(y_train_idx, num_classes=num_classes)
                y_test_final = to_categorical(y_test_idx, num_classes=num_classes)
            else:
                y_train_final = y_train_idx
                y_test_final = y_test_idx
            # ---------------------------

            # select model (auto chooses appropriate output layer based on num_classes)
            model, model_name = auto_model_selector.select_model("tabular", input_shape=input_shape, num_classes=num_classes)
            print(f"[ExplainDL] Selected model {model_name}")

            # optional tuning (example: tabular tuner)
            if enable_tuning:
                try:
                    best_model = tuner.tune_tabular_model(X_train.values, y_train_final, input_shape=input_shape, num_classes=num_classes, max_trials=max_trials)
                    model = best_model
                    model_name += " (tuned)"
                except Exception as e:
                    print(f"[ExplainDL] Tuning failed: {e}")

            # train
            trained = train_and_evaluate(model, "tabular", (X_train.values, X_test.values, y_train_final, y_test_final), epochs=5, batch_size=32)

            # explainability (SHAP + LIME)
            if show_explainability:
                # SHAP
                try:
                    # create a DataFrame sample for SHAP if we have column names, otherwise use array
                    if isinstance(X_train, pd.DataFrame):
                        X_sample = X_train.sample(n=min(100, len(X_train)))
                    else:
                        X_sample = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])]).sample(n=min(100, X_train.shape[0]))

                    shap_fig = shap_explainer.explain_with_shap(model, X_sample)
                    shap_path = os.path.join(temp_dir, "explain_shap.png")
                    _save_matplotlib_figure(shap_fig, shap_path)
                    explain_fig_path = shap_path
                except Exception as e:
                    print(f"[ExplainDL] SHAP explain failed: {e}")
                    shap_fig = None
                    explain_fig_path = None

                # LIME (local)
                try:
                    feature_names = list(X_sample.columns) if isinstance(X_sample, pd.DataFrame) else [f"f{i}" for i in range(X_train.shape[1])]
                    X_train_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
                    X_instance = X_test[0] if hasattr(X_test, "__len__") else X_test

                    lime_res = lime_explainer.explain_with_lime(model, X_train_arr, X_instance, feature_names)
                    # lime_explainer may return (fig, df) or fig only depending on implementation
                    if isinstance(lime_res, tuple):
                        lime_fig, lime_df = lime_res
                    else:
                        lime_fig = lime_res
                        lime_df = None

                    lime_path = os.path.join(temp_dir, "explain_lime.png")
                    _save_matplotlib_figure(lime_fig, lime_path)
                    # prefer SHAP image as primary explainability display; fallback to LIME
                    if not explain_fig_path:
                        explain_fig_path = lime_path
                except Exception as e:
                    print(f"[ExplainDL] LIME explain failed: {e}")
                    lime_fig = None

        elif data_type == "image":
            # images: extract
            extracted_dir = image_loader.extract_image_dataset(saved_path, extract_dir=os.path.join(temp_dir, "images"))
            valid_count = image_loader.verify_images(extracted_dir)
            if valid_count == 0:
                raise ValueError("No valid images found in the provided ZIP.")

            # preprocessing -> generators (fixed: proper split and class_mode)
            train_gen, val_gen = image_preprocess.preprocess_image_data(
                extracted_dir,
                target_size=(128, 128),
                batch_size=16,
                val_split=0.2,
                seed=42
            )

            # debug info (optional - helpful during test)
            print(f"[ExplainDL] Image dataset: {train_gen.samples} train / {val_gen.samples} val")
            print(f"[ExplainDL] Class indices: {train_gen.class_indices}")

            # determine num_classes from generator
            num_classes = getattr(train_gen, "num_classes", None) or len(getattr(train_gen, "class_indices", {}))
            input_shape = (128, 128, 3)

            # select model (ensures correct output layer)
            model, model_name = auto_model_selector.select_model("image", input_shape=input_shape, num_classes=num_classes)
            print(f"[ExplainDL] Selected model {model_name} (classes={num_classes})")

            # compute steps per epoch to avoid incomplete batches issues
            steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
            validation_steps = max(1, val_gen.samples // val_gen.batch_size)

            # train (use slightly more epochs; trainer has EarlyStopping)
            trained = train_and_evaluate(
                model,
                "image",
                (train_gen, val_gen),
                epochs=8,
                batch_size=16
            )

            # Grad-CAM for a single sample
            if show_explainability:
                try:
                    # get one validation batch (val_gen was created with shuffle=False)
                    sample_batch, sample_labels = next(val_gen)
                    img_array = sample_batch[:1]
                    grad_fig = gradcam_explainer.generate_gradcam(model, img_array)
                    grad_path = os.path.join(temp_dir, "explain_gradcam.png")
                    _save_matplotlib_figure(grad_fig, grad_path)
                    explain_fig_path = grad_path
                except Exception as e:
                    print(f"[ExplainDL] Grad-CAM failed: {e}")
                    grad_fig = None
                    explain_fig_path = None


        elif data_type == "text":
            # load raw lines
            lines = text_loader.load_text_data(saved_path)

            # parse label+text lines
            texts, labels = _parse_text_lines_to_xy(lines)

            # clean + tokenize using existing helper functions
            from explainDL.preprocessing.text_preprocess import clean_text
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            cleaned = [clean_text(t) for t in texts]

            vocab_size = 10000
            max_len = 100
            tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
            tokenizer.fit_on_texts(cleaned)
            sequences = tokenizer.texts_to_sequences(cleaned)
            padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

            # labels -> numeric
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y = np.array([label_map[v] for v in labels])
            if num_classes > 2:
                y_final = to_categorical(y, num_classes=num_classes)
            else:
                y_final = y

            X_train, X_test, y_train, y_test = train_test_split(padded, y_final, test_size=0.2, random_state=42)

            # select model (for text, input_shape = (vocab_size, max_len) as used earlier)
            input_shape = (vocab_size, max_len)
            model, model_name = auto_model_selector.select_model("text", input_shape=input_shape, num_classes=num_classes)
            print(f"[ExplainDL] Selected model {model_name}")

            # train
            trained = train_and_evaluate(model, "text", (X_train, X_test, y_train, y_test), epochs=3, batch_size=32)

            # LIME or SHAP for text can be expensive — try LIME per-instance
            if show_explainability:
                try:
                    # use a small example: explain first instance
                    X_train_for_lime = X_train if isinstance(X_train, np.ndarray) else np.array(X_train)
                    X_instance = X_test[0]
                    # LIME's tabular explainer expects numeric arrays; use token counts feature names
                    feature_names = [f"tok{i}" for i in range(X_instance.shape[0])]
                    lime_res = lime_explainer.explain_with_lime(model, X_train_for_lime, X_instance, feature_names)
                    if isinstance(lime_res, tuple):
                        lime_fig, _ = lime_res
                    else:
                        lime_fig = lime_res
                    explain_fig_path = os.path.join(temp_dir, "explain_text_lime.png")
                    _save_matplotlib_figure(lime_fig, explain_fig_path)
                except Exception as e:
                    print(f"[ExplainDL] Text LIME failed: {e}")
                    lime_fig = None
                    explain_fig_path = None
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Collect metrics and prepare report
        if trained:
            metrics = trained.get("metrics", {})
            summary_df = trained.get("summary")
            results["metrics"] = summary_df if summary_df is not None else pd.DataFrame()
            results["raw_metrics"] = metrics
            results["model_name"] = model_name

            # decide primary explainability image
            if show_explainability and explain_fig_path and os.path.exists(explain_fig_path):
                results["explainability_fig"] = explain_fig_path
            else:
                results["explainability_fig"] = None

            # generate PDF report if possible
            try:
                report_path = report_generator.generate_report(results["metrics"],
                                                              metrics_dict=results.get("raw_metrics"),
                                                              shap_fig=shap_fig,
                                                              lime_fig=lime_fig,
                                                              gradcam_fig=grad_fig,
                                                              output_path=os.path.join(temp_dir, "ExplainDL_Report.pdf"))
                results["report_path"] = report_path
            except Exception as e:
                print(f"[ExplainDL] Report generation failed: {e}")
                results["report_path"] = None

        else:
            raise RuntimeError("Training did not complete successfully.")

        return results

    except Exception as e:
        # Bubble up an informative error
        raise RuntimeError(f"ExplainDL pipeline failed: {e}")

    finally:
        # Note: we intentionally keep temp_dir for debugging if you want to inspect outputs.
        # If you'd like to remove it automatically, uncomment the following:
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass
