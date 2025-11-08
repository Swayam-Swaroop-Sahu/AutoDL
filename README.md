# 🧠 ExplainDL: Automated Deep Learning Analysis Tool

**Author:** Swayam Swaroop Sahu  
**Institution:** VIT Vellore  
**Department:** Computer Science and Engineering  
**Project Type:** Academic Research Project (MVP Stage)  
**Semester:** 5th Semester, 2025  

---

## 📘 Overview

**ExplainDL** is an **Automated Deep Learning Analysis Tool** designed to make deep learning accessible to non-technical users.  
It automates the entire AI workflow — from **dataset upload to preprocessing, model training, evaluation, and explainability** — without requiring manual coding.

Users can upload their datasets (tabular, image, or text), and ExplainDL automatically:
- Detects the dataset type.  
- Cleans and preprocesses the data.  
- Selects and trains a suitable deep learning model.  
- Evaluates performance metrics (Accuracy, F1-score, etc.).  
- Generates **Explainability visualizations** using SHAP, LIME, and Grad-CAM.  
- Produces a **comprehensive analysis report** in PDF format.  

ExplainDL combines **AutoML (automation)** and **XAI (explainable AI)** principles in one modular system.

---

## 🎯 Key Features

- ✅ Automatic dataset type detection (tabular / image / text).  
- ⚙️ End-to-end data preprocessing.  
- 🧩 Dynamic model selection:
  - MLP for tabular data  
  - CNN for image data  
  - LSTM for text data  
- 🔍 Model training and evaluation (accuracy, precision, recall, F1-score).  
- 🧠 Explainability integration using SHAP, LIME, and Grad-CAM.  
- 🧾 Auto-generated PDF analysis reports.  
- 🎛️ Optional hyperparameter tuning via **Keras-Tuner**.  
- 🖥️ Streamlit web interface for easy, no-code operation.  

---

## 🏗️ Project Structure

```bash
ExplainDL/
│
├── app.py                     # Streamlit front-end app
│
├── explainDL/
│   ├── __init__.py
│   │
│   ├── data_input/            # Handles data upload and type detection
│   │   ├── detect_type.py
│   │   ├── tabular_loader.py
│   │   ├── image_loader.py
│   │   └── text_loader.py
│   │
│   ├── preprocessing/         # Data cleaning & preprocessing
│   │   ├── tabular_preprocess.py
│   │   ├── image_preprocess.py
│   │   └── text_preprocess.py
│   │
│   ├── model_selection/       # Auto model selection logic
│   │   ├── auto_model_selector.py
│   │   ├── tabular_models.py
│   │   ├── image_models.py
│   │   └── text_models.py
│   │
│   ├── training/              # Model training and evaluation
│   │   └── trainer.py
│   │
│   ├── explainability/        # Explainability modules
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── gradcam_explainer.py
│   │   └── report_generator.py
│   │
│   ├── utils/                 # Helper functions
│   │   ├── file_utils.py
│   │   └── visualization_utils.py
│   │
│   └── pipeline.py            # Core end-to-end orchestrator
│
├── requirements.txt           # All dependencies
└── README.md                  # Project documentation (this file)



---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ExplainDL.git
cd ExplainDL

2. Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate        # for Linux/Mac
venv\Scripts\activate           # for Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
streamlit run app.py


The app will start locally.
You can access it via: http://localhost:8501

📂 Supported Data Formats
Data Type	Supported File Format	Example
Tabular	.csv, .xlsx	Medical / Financial data
Image	.zip (with subfolders per class)	Cats vs Dogs, Human vs Object
Text	.txt (each line: label,text)	Sentiment dataset
📊 Output and Results

After running the automated pipeline, ExplainDL generates:

Performance Metrics:

Accuracy, Precision, Recall, F1-score

Confusion Matrix & per-class classification report

Explainability Visuals:

SHAP: Feature importance (tabular)

LIME: Instance-level feature influence

Grad-CAM: Heatmaps highlighting key regions in images

Auto-generated PDF Report

Includes metrics summary, visuals, and model insights

🧪 Example Usage

Open the Streamlit web app.

Upload your dataset (.csv, .zip, or .txt).

Configure options in the sidebar:

Auto Mode (ON by default)

Show Explainability Report (ON/OFF)

Enable Hyperparameter Tuning (Optional)

Click Start Analysis.

Wait for the pipeline to complete.

View model metrics and download the PDF report.

🧠 Technologies Used
Category	Tools / Frameworks
Language	Python
Frontend	Streamlit
Deep Learning	TensorFlow, Keras
Explainability (XAI)	SHAP, LIME, Grad-CAM
Data Handling	Pandas, NumPy, Scikit-learn
Hyperparameter Tuning	Keras-Tuner
Visualization	Matplotlib, Seaborn
Documentation	Markdown, FPDF

🧩 Current Limitations

Accuracy on small datasets may be low due to limited training data.

Grad-CAM sometimes fails for single-class image datasets.

Models are not yet exported after training.

Performance tuning is basic; can be expanded using advanced search spaces.

🚀 Future Enhancements

Cloud deployment for remote model execution.

Model export functionality (.h5 / .onnx).

Support for multi-label and regression tasks.

Integration of advanced models (Transformers, EfficientNetV2).

Enhanced visual dashboards for explainability.

Comparative benchmarking against public datasets.

Potential patent filing for the combined AutoML + XAI workflow.

📚 References

This project draws inspiration from:

AutoML frameworks: AutoKeras
, Auto-Sklearn

Explainability libraries: SHAP
, LIME
, Grad-CAM

TensorFlow & Keras official documentation

Research papers in Explainable AI and Automated Machine Learning

🧾 License

This project is intended for educational and research purposes only.
© 2025 Swayam Swaroop Sahu. All rights reserved.
