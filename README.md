# ExplainDL: Automated Deep Learning Analysis Tool

**Author:** Swayam Swaroop Sahu  
**Institution:** VIT Vellore  
**Department:** Computer Science and Engineering  
**Project Type:** Academic Research Project (MVP Stage)  
**Semester:** 5th Semester, 2025 

---

### **OVERVIEW**

ExplainDL is an **Automated Deep Learning Analysis Tool** designed to make deep learning accessible to non-technical users.
It automates the complete AI workflow — from **dataset upload to preprocessing, model training, evaluation, and explainability** — without requiring manual coding.

Users can upload datasets (tabular, image, or text), and ExplainDL automatically:

* Detects the dataset type
* Cleans and preprocesses data
* Selects and trains an appropriate deep learning model
* Evaluates performance metrics (Accuracy, Precision, Recall, F1-score)
* Generates explainability visualizations using **SHAP**, **LIME**, and **Grad-CAM**
* Produces a detailed **analysis report (PDF)**

ExplainDL combines **AutoML (Automation)** and **XAI (Explainable AI)** into one modular system.

---

### **KEY FEATURES**

* Automatic dataset type detection (Tabular / Image / Text)
* End-to-end preprocessing for all supported formats
* Dynamic model selection:
  • MLP for tabular data
  • CNN for image data
  • LSTM for text data
* Automated training and evaluation pipeline
* Integrated explainability layer (SHAP, LIME, Grad-CAM)
* PDF report generation for results and analysis
* Optional hyperparameter tuning using **Keras-Tuner**
* Streamlit-based front-end for a user-friendly interface

---

### **PROJECT STRUCTURE**

```
ExplainDL/
│
├── app.py                     → Streamlit front-end app
│
├── explainDL/
│   ├── data_input/            → Dataset loading and type detection
│   ├── preprocessing/         → Cleaning and preprocessing pipelines
│   ├── model_selection/       → Auto model selection (MLP/CNN/LSTM)
│   ├── training/              → Model training and evaluation logic
│   ├── explainability/        → Explainability modules (SHAP, LIME, Grad-CAM)
│   ├── utils/                 → Helper functions and utilities
│   └── pipeline.py            → Main orchestrator for the full workflow
│
├── requirements.txt           → Dependency list
└── README.txt / README.md     → Project documentation
```

---

### **INSTALLATION AND SETUP**

**Step 1:** Clone the Repository

```
git clone https://github.com/<your-username>/ExplainDL.git
cd ExplainDL
```

**Step 2:** Create and Activate Virtual Environment

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)
```

**Step 3:** Install Dependencies

```
pip install -r requirements.txt
```

**Step 4:** Run the Application

```
streamlit run app.py
```

Access the app locally through:
**[http://localhost:8501](http://localhost:8501)**

---

### **SUPPORTED DATA FORMATS**

| **Data Type** | **Supported Format**          | **Example**                   |
| ------------- | ----------------------------- | ----------------------------- |
| Tabular       | .csv, .xlsx                   | Medical or financial records  |
| Image         | .zip (with folders per class) | Cats vs Dogs, Human vs Object |
| Text          | .txt (label,text per line)    | Sentiment analysis data       |

---

### **OUTPUT AND RESULTS**

After execution, ExplainDL generates the following outputs:

1. **Performance Metrics**

   * Accuracy, Precision, Recall, F1-score
   * Confusion Matrix
   * Per-class classification report

2. **Explainability Visuals**

   * SHAP: Feature importance for tabular data
   * LIME: Local interpretability for individual samples
   * Grad-CAM: Visual heatmaps for image classification

3. **PDF Report**

   * Includes all metrics, visuals, and interpretation summary

---

### **EXAMPLE USAGE**

1. Launch the Streamlit app.
2. Upload dataset (.csv, .zip, or .txt).
3. Configure sidebar options:

   * Auto Mode (default ON)
   * Show Explainability Report (ON/OFF)
   * Enable Hyperparameter Tuning (Optional)
4. Click **Start Analysis**.
5. Wait for the automated pipeline to complete.
6. View model performance and download the report.

---

### **TECHNOLOGIES USED**

**Programming Language:** Python
**Frontend Interface:** Streamlit
**Deep Learning Framework:** TensorFlow, Keras
**Explainability Libraries:** SHAP, LIME, Grad-CAM
**Data Handling:** Pandas, NumPy, Scikit-learn
**Hyperparameter Tuning:** Keras-Tuner
**Visualization:** Matplotlib, Seaborn
**Reporting:** FPDF

---

### **CURRENT LIMITATIONS**

* Accuracy may vary for small datasets.
* Grad-CAM can fail when dataset contains only one class.
* Trained model export feature not yet implemented.
* Limited hyperparameter search scope (basic tuner setup).

---

### **FUTURE ENHANCEMENTS**

* Cloud deployment for larger datasets.
* Model export functionality (.h5 / .onnx).
* Multi-label and regression problem support.
* Integration of advanced models (Transformers, EfficientNetV2).
* Improved explainability dashboard (interactive visualizations).
* Comparative benchmarking with public datasets.
* Patent exploration for combined AutoML + XAI design.

---

### **REFERENCES**

The system builds upon principles and libraries from:

* AutoML frameworks: AutoKeras, Auto-Sklearn, H2O.ai, Google AutoML
* Explainability frameworks: SHAP, LIME, Grad-CAM
* TensorFlow and Keras documentation
* Research papers on Explainable AI (XAI) and Automated Machine Learning

---

### **LICENSE**

This project is developed for **educational and research purposes only**.
© 2025 Swayam Swaroop Sahu. All Rights Reserved.

---

