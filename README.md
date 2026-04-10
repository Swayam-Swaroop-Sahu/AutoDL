**ExplainDL - Automated Classification and Explainability System**
=======================================================

**Author:** Swayam Swaroop Sahu\
**Institution:** VIT Vellore\
**Department:** Computer Science and Engineering\
**Project Type:** Academic Research Project (MVP Stage)

* * * * *

**OVERVIEW**
---------------

**ExplainDL** is an **Automated Classification and Explainability System** built to make AI accessible even to non-technical users.\
It automates the full ML pipeline:

-   **Dataset Upload → Preprocessing → Model Selection → Training → Evaluation → Explainability → Report Generation**

ExplainDL supports **Tabular**, **Image**, and **Text datasets**, and intelligently:

-   Detects dataset type

-   Preprocesses data

-   Selects the optimal model dynamically (MLP / CNN / LSTM / BiLSTM / Text-CNN)

-   Trains and evaluates models

-   Computes full metrics (Accuracy, Precision, Recall, F1-score)

-   Generates **Explainability outputs** (Confusion Matrix, SHAP, LIME, Grad-CAM, prediction distribution)



-   Supports **Hyperparameter Tuning** with Keras-Tuner

This project combines automated deep learning workflow orchestration with explainability features in a modular framework.

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

**New Features:**
- **Model Comparison Dashboard**: See all candidate models with scores, descriptions, pros/cons
- **Manual Override Mode**: Choose a specific model architecture if you prefer
- **Selection Explanations**: Understand why a particular model was selected
- **Training Explanations**: Get human-readable explanations of the training process
- **Prediction Explanations**: Understand what the results mean in plain language

### **Performance & Explainability**

-   **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report

-   **Visualizations**: Loss/Accuracy curves, Confusion Matrix, Prediction Distribution (Histogram + Pie Chart)

-   **Multi-Layer Explainability**:
    - **Model Selection**: Why a model was chosen, comparison of all candidates
    - **Training Phase**: How training progressed, what metrics mean, data-specific insights
    - **Prediction Phase**: What predictions mean, how the model made decisions, actionable insights

-   **Human-Readable Explanations**: Text-based explanations for all stages (selection, training, prediction)



### **Hyperparameter Tuning (Optional)**

Powered by **Keras-Tuner RandomSearch**

-   Tunable layers, units, dropout, learning rate, embedding size

-   Works for tabular, image, and text pipelines

### **User-Friendly UI (Streamlit)**

-   Upload datasets with automatic validation

-   **Model Selection**: Choose automatic or manual model selection

-   Select tuning options

-   View comprehensive metrics, graphs, and explanations

-   **Real-time Feedback**: See model comparison, selection reasoning, and training progress

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

**INSTALLATION & SETUP**
---------------------------

### **1 Clone the Repository**

`git clone https://github.com/Swayam-Swaroop-Sahu/ExplainDL`\
`cd ExplainDL`

### **2 Create & Activate Environment (Optional - pip)**

`python -m venv venv`
        
Windows:\
`venv\Scripts\activate`

Mac/Linux:\
`source venv/bin/activate`

### **3 Install Dependencies**

Using uv (Recommended):

`pip install uv`\
`uv add -r requirements.txt`

Using pip:

`pip install -r requirements.txt`

### **4 Run ExplainDL**

Using uv:

`uv run streamlit run app.py`

Using pip:

`streamlit run app.py`

### **Access Locally**

Open in Browser:

**<http://localhost:8501>**

* * * * *

**SUPPORTED DATA FORMATS**
-----------------------------

| Type | Format | Example Use-Case | Requirements |
| --- | --- | --- | --- |
| Tabular | `.csv`, `.xlsx` | Finance, medical records | Min 10 rows, 2 columns (including target) |
| Image | `.zip` (folders = classes) | Cats vs Dogs, defect detection | Min 10 images, organized in class folders |
| Text | `.txt` (`label <TAB> text`) | Sentiment analysis | Format: `label<TAB>text` or `label,text` per line |

**Note**: ExplainDL includes comprehensive validation and error handling for incorrect dataset formats. Users receive clear error messages with guidance on how to fix issues.

* * * * *

**OUTPUT & RESULTS**
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



* * * * *

**EXAMPLE USAGE FLOW**
-------------------------

1.  Run the Streamlit app: `streamlit run app.py`

2.  Upload labelled dataset (CSV/XLSX for tabular, ZIP for images, TXT for text)

3.  **Choose Model Selection Mode**:
    -   Automatic (Recommended): System selects the most suitable model
    -   Manual Override: Select specific model architecture

4.  Optionally enable hyperparameter tuning

5.  Train the model

6.  **View Results**:
    -   Model selection details and comparison
    -   Training metrics and visualizations
    -   Explanations (selection, training process)



8.  Upload unlabelled dataset and run predictions

9.  **View Prediction Results**:
    -   Predictions table
    -   Visualizations (histogram, pie chart)
    -   Explanation text



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


* * * * *

**CURRENT LIMITATIONS**
--------------------------

-   Accuracy may vary for very small datasets (< 50 samples)

-   Grad-CAM may fail with single-class datasets

-   Advanced explainability dashboards not yet integrated

-   Hyperparameter search is intentionally shallow for speed

-   No ONNX export yet

**Error Handling**
-   Comprehensive validation for all dataset types
-   Clear error messages for incorrect formats
-   Automatic detection and reporting of data quality issues

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
