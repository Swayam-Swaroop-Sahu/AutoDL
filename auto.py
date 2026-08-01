#!/usr/bin/env python
"""AutoDL CLI — train and predict from the command line.

Usage:
  # Train with explicit target column
  python auto.py train --data my_data.csv --target survived

  # Train with interactive target selection (shows ranked table, prompts user)
  python auto.py train --data my_data.csv

  # Predict using a previously trained model
  python auto.py predict --data unlabelled.csv --model <model_id>
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for direct invocation
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _show_ranked_table(ranked: list[dict]) -> None:
    """Pretty-print the ranked target candidates table to stderr/stdout."""
    print("\n─── Ranked target candidates (higher score = more likely target) ───")
    header = f"{'Rank':<5} {'Column':<25} {'Score':<8} {'Name':<8} {'Card':<8} {'# Unique':<10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<5} {r['col']:<25} {r['score']:<8.4f} "
            f"{r['name_score']:<8.2f} {r['card_score']:<8.2f} {r['n_unique']:<10}"
        )
    print("-" * len(header))
    top = ranked[0]
    print(f"Top suggestion: '{top['col']}'  (score={top['score']:.4f})")


def prompt_target(ranked: list[dict]) -> str:
    """Interactive prompt for user to pick a target column."""
    print(
        "\nAutoDL requires you to confirm the target column. "
        "Never auto-selects silently."
    )
    for i, r in enumerate(ranked, 1):
        print(f"  [{i}] {r['col']}  (score={r['score']:.4f})")
    print(f"  [{len(ranked) + 1}]  Type a column name manually")
    while True:
        choice = input(f"\nSelect [1–{len(ranked) + 1}]: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(ranked):
                return ranked[idx - 1]["col"]
            if idx == len(ranked) + 1:
                custom = input("Column name: ").strip()
                if custom:
                    return custom
        except ValueError:
            pass
        print(f"Enter a number between 1 and {len(ranked) + 1}.")


def cmd_train(args):
    """Run the full training pipeline."""
    from src.core.pipeline_train import train_pipeline

    if not os.path.exists(args.data):
        print(f"Error: file not found — {args.data}")
        return 1

    target_col = args.target
    if target_col is None:
        # Auto-detect + ranked table + interactive prompt
        import pandas as pd
        from src.data.tabular_loader import load_tabular_data
        from src.target_detection import rank_target_candidates

        df = load_tabular_data(args.data, require_target=True)
        ranked = rank_target_candidates(df)
        _show_ranked_table(ranked)
        target_col = prompt_target(ranked)

    if target_col is None:
        print("Error: no target column provided. Use --target <col>.")
        return 1

    print(f"Using target column: '{target_col}'")
    print("Starting training...")
    try:
        result = train_pipeline(args.data, target_col=target_col)
        print(f"✅ Training complete! Model ID: {result['model_id']}")
        print(f"   Model: {result.get('model_name', '?')}")
        print(f"   Type:  {result['dataset_type']}")
        print(f"   Dir:   {result['model_dir']}")
        print(f"Use this ID to predict: --model {result['model_id']}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    return 0


def cmd_predict(args):
    """Run prediction with a trained model."""
    from src.core.pipeline_predict import predict_pipeline

    if not args.model:
        print("Error: --model <model_id> is required for prediction.")
        return 1
    if not os.path.exists(args.data):
        print(f"Error: prediction file not found — {args.data}")
        return 1

    model_dir = os.path.join("model_registry", args.model)
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found — {model_dir}")
        print("  Run `auto.py train` first.")
        return 1

    try:
        result = predict_pipeline(model_dir, args.data)
        preds = result.get("predictions", [])
        labels = result.get("prediction_labels", preds)

        print("\n── Predictions ──")
        for i, (p, lbl) in enumerate(zip(preds[:20], labels[:20])):
            print(f"  [{i}] class={p} ({lbl})")
        if len(preds) > 20:
            print(f"  ... and {len(preds) - 20} more")

        # Save to CSV next to input
        import json
        out_path = args.data + ".predictions.csv"
        with open(out_path, "w") as f:
            f.write("index,predicted_class,label\n")
            for i, (p, lbl) in enumerate(zip(preds, labels)):
                f.write(f"{i},{p},{lbl}\n")
        print(f"Wrote predictions to: {out_path}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AutoDL — automated classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auto.py train --data labelled.csv --target survived
  auto.py train --data labelled.csv      (interactive column pick)
  auto.py predict --data test.csv --model a1b2c3d4
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train", help="Train a classification model")
    train_p.add_argument("--data", required=True, help="Path to labelled dataset (.csv, .xlsx, .txt, .zip)")
    train_p.add_argument(
        "--target", default=None,
        help="Target column name. If omitted, you'll be shown a ranked list to pick from.",
    )

    # predict
    pred_p = sub.add_parser("predict", help="Predict using a trained model")
    pred_p.add_argument("--data", required=True, help="Path to unlabelled data")
    pred_p.add_argument(
        "--model", required=True,
        help="Model ID (the short hex id shown after training)",
    )

    args = parser.parse_args()

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())