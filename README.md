**ExplainDL: Automated Deep Learning Analysis Tool**
=======================================================

**Author:** Swayam Swaroop Sahu\
**Institution:** VIT Vellore\
**Department:** Computer Science and Engineering\
**Project Type:** Academic Research Project (MVP Stage)\
**Semester:** 5th Semester, 2025

* * * * *

**OVERVIEW**
---------------

**ExplainDL** is an **Automated Deep Learning Analysis Tool** built to make AI accessible even to non-technical users.\
It automates the full ML pipeline:

-   **Dataset Upload → Preprocessing → Model Selection → Training → Evaluation → Explainability → Report Generation**

ExplainDL supports **Tabular**, **Image**, and **Text datasets**, and intelligently:

-   Detects dataset type

-   Preprocesses data

-   Selects the optimal model dynamically (MLP / CNN / LSTM / BiLSTM / Text-CNN)

-   Trains and evaluates models

-   Computes full metrics (Accuracy, Precision, Recall, F1-score)

-   Generates **Explainability outputs** (Confusion Matrix, SHAP, LIME, Grad-CAM, prediction distribution)

-   Produces a **detailed PDF report**

-   Supports **Hyperparameter Tuning** with Keras-Tuner

This project merges **AutoML + Explainable AI (XAI)** into one modular, extendable framework.

* * * * *

**KEY FEATURES**
-------------------

### **Dataset Intelligence**

-   Automatic dataset type detection

-   Tabular (.csv, .xlsx), Image (.zip), Text (.txt with `label<TAB>text`)

### **Preprocessing Pipelines**

-   Scaling, encoding, tokenization, augmentation depending on data type

### **Automated Model Selection**

| Data Type | Model Options |
| --- | --- |
| **Tabular** | MLP-Small, Medium, Large |
| **Image** | Small-CNN, MobileNetV2, EfficientNetB0 |
| **Text** | LSTM, BiLSTM, Text-CNN |

ExplainDL performs a forward-pass variance check to pick the best model.

### **Performance & Explainability**

-   Accuracy, Precision, Recall, F1-score

-   Confusion Matrix

-   Classification Report

-   Prediction Distribution (Histogram + Pie Chart)

-   Non-technical explanation text

-   Auto-generated PDF report

### **Hyperparameter Tuning (Optional)**

Powered by **Keras-Tuner RandomSearch**

-   Tunable layers, units, dropout, learning rate, embedding size

-   Works for tabular, image, and text pipelines

### **User-Friendly UI (Streamlit)**

-   Upload datasets

-   Select tuning options

-   View metrics, graphs, explanations

-   Download reports and predictions

* * * * *

**PROJECT STRUCTURE**
------------------------
<pre>
ExplainDL/
│
├── app.py                        → Streamlit front-end UI
│
├── explainDL/
│   ├── data/                     → Data loading, extraction, type detection
│   ├── preprocessing/            → Tabular, image, text preprocessors
│   ├── model_selection/          → Auto model selector + hyperparameter tuner
│   ├── training/                 → Model training + metrics
│   ├── explainability/           → Reports, SHAP/LIME/Grad-CAM utilities
│   ├── core/                     → Pipeline orchestrators for training/prediction
│   ├── registry.py               → Model registry handler
│   └── utils/                    → Helper utilities
│
├── requirements.txt              → Dependency list
└── README.md                     → Project documentation
</pre>

* * * * *

⚙️ **INSTALLATION & SETUP**
---------------------------

### **1 Clone the Repository**

`git clone https://github.com/<your-username>/ExplainDL.git
cd ExplainDL`

### **2 Create & Activate Environment**

`python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux`

### **3 Install Dependencies**

`pip install -r requirements.txt`

### **4 Run ExplainDL**

`streamlit run app.py`

Access locally at:\
**<http://localhost:8501>**

* * * * *

**SUPPORTED DATA FORMATS**
-----------------------------

| Type | Format | Example Use-Case |
| --- | --- | --- |
| Tabular | `.csv`, `.xlsx` | Finance, medical records |
| Image | `.zip` (folders = classes) | Cats vs Dogs, defect detection |
| Text | `.txt` (`label <TAB> text`) | Sentiment analysis |

* * * * *

📤 **OUTPUT & RESULTS**
-----------------------

### **1\. Performance Metrics**

-   Accuracy

-   Precision

-   Recall

-   F1-score

-   Confusion Matrix

-   Per-class breakdown

### **2\. Deep Explainability**

-   **Confusion Matrix heatmap**

-   **Prediction distribution** (histogram + pie chart)

-   **SHAP / LIME / Grad-CAM hooks (architecture-ready)**

-   **Human-readable explanation text**

### **3\. PDF Report (Auto-Generated)**

Contains:

-   Summary of metrics

-   All graphs

-   Confusion matrix

-   Classification report

-   Simple non-technical explanation

* * * * *

**EXAMPLE USAGE FLOW**
-------------------------

1.  Run the Streamlit app

2.  Upload labelled dataset

3.  Optionally enable hyperparameter tuning

4.  Train the model

5.  Download:

    -   Predictions CSV

    -   Training Report PDF

6.  Upload unlabelled dataset and run predictions

7.  View explainability graphs and text summary

* * * * *

**TECH STACK**
-----------------

| Component | Technology |
| --- | --- |
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Explainability | SHAP, LIME, Grad-CAM |
| Tuning | Keras-Tuner |
| UI | Streamlit |
| Data | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Reporting | FPDF |

* * * * *

**CURRENT LIMITATIONS**
--------------------------

-   Accuracy may vary for small datasets

-   Grad-CAM may fail with single-class datasets

-   Advanced explainability dashboards not yet integrated

-   Hyperparameter search is intentionally shallow for speed

-   No ONNX export yet

* * * * *

**FUTURE ENHANCEMENTS**
--------------------------

-   Cloud-based training & GPU compute

-   Advanced model zoo (Transformers, EfficientNetV2, ViT)

-   Interactive SHAP dashboards

-   Multi-label + regression support

-   ONNX export

-   AutoML search space expansion

-   Patent exploration for ExplainDL architecture

* * * * *

**REFERENCES**
-----------------

ExplainDL builds upon ideas from:

-   AutoML: AutoKeras, Auto-Sklearn, H2O.ai

-   XAI Tools: SHAP, LIME, Grad-CAM

-   TensorFlow / Keras documentation

-   Research papers on XAI & Responsible AI

* * * * *

**LICENSE**
--------------

This project is created for **academic and research purposes only**.\
© 2025 **Swayam Swaroop Sahu** --- All Rights Reserved.