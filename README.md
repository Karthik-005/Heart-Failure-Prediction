# Heart Failure Prediction Project

## Overview

This project aims to predict the likelihood of heart failure in patients based on a set of 11 clinical features. The goal is to develop a robust machine learning model that can serve as a valuable tool for early-stage detection. The dataset used is the "Heart Failure Prediction" dataset, sourced from Kaggle.

This project follows a structured machine learning workflow, from data exploration and preprocessing to model training, hyperparameter tuning, and evaluation.

---

## Project Structure

The repository is organized to ensure clarity and reproducibility:

```
heart-failure-prediction/
│
├── data/
│   └── heart.csv              # The raw, original dataset
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # In-depth data analysis and visualization
│   └── 02_modeling_and_evaluation.ipynb  # Preprocessing, model training, and evaluation
│
├── models/
│   ├── random_forest.pkl
│   └── xgb.pkl
|
├── requirements.txt           # A list of all necessary Python libraries
│
└── README.md                  # This project overview file
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
The first phase involved a deep dive into the dataset to understand the characteristics and distributions of each feature. Key steps included:
* **Data Inspection:** Initial review of data types, shapes, and summary statistics.
* **Missing Value Analysis:** Identified that `Cholesterol` had a significant number of `0` values, which were treated as missing data. The final strategy adopted was to drop these rows.
* **Visualization:** Used histograms, box plots, and correlation heatmaps to understand feature distributions, identify outliers, and analyze relationships between variables and the target (`HeartDisease`).

### 2. Preprocessing and Feature Engineering
A `scikit-learn` pipeline was constructed to create a reproducible and robust workflow, preventing data leakage. The pipeline automates the following steps for any raw data fed into it:
* **Numerical Features:** Missing values are imputed using the median, and features are scaled using `StandardScaler`.
* **Categorical Features:** Missing values are imputed using the most frequent value, and features are converted into a numerical format using `OneHotEncoder`.

### 3. Model Training and Hyperparameter Tuning
Several machine learning models were trained and compared to find the best performer:
* **Random Forest Classifier**
* **XGBoost Classifier** (with GPU acceleration)
* **Support Vector Classifier (SVC)**
* **Logistic Regression** (as a baseline)

For each model, `GridSearchCV` was used to perform an exhaustive search for the optimal hyperparameters, ensuring each model was trained to its full potential.

### 4. Model Evaluation
Models were evaluated on a held-out test set. Given the medical context, the evaluation focused on more than just accuracy:
* **Classification Report:** Provided detailed **precision**, **recall**, and **F1-scores**. Recall for the positive class (predicting actual heart disease) was considered a critical metric to minimize missed diagnoses.
* **Confusion Matrix:** Visualized the model's performance in terms of True Positives, True Negatives, False Positives, and False Negatives.

---

## Results

After a comprehensive tuning and evaluation process, the models performed as follows. The **[Your Best Model Name]** was selected as the final model due to its superior performance, particularly in identifying patients with heart disease.

| Model | Test Accuracy | F1-Score (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: |
| <!--Logistic Regression | *[Enter Score]* | *[Enter Score]* | *[Enter Score]* |-->
| RandomForest | 87.33% | 0.87 | 0.85 |
| XGBoost | 85.33% | 0.84 | 0.80 |
| SVC | 84.67% | 0.84 | 0.81 |

**Final Model:** The best performing model, **Random Forest Classifier**, was saved to the `/models` directory along with all other models that were trained.

---

## How to Run This Project

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Karthik-005/Heart-Failure-Prediction.git](https://github.com/Karthik-005/Heart-Failure-Prediction.git)
    cd heart-failure-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebooks:**
    Launch Jupyter Lab or Jupyter Notebook and navigate through the notebooks in the `notebooks/` directory.
    ```bash
    jupyter notebook
    ```

---
### **Action Item for You**
Before you commit this file, make sure to create the `requirements.txt` file by running this command in your terminal:
```bash
pip freeze > requirements.txt
```
This will save all the libraries and their specific versions needed to run your project, making it fully reproducible.
