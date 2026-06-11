# 📉 Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using the **Telco Customer Churn** dataset. Built with XGBoost and Random Forest classifiers, with full preprocessing and evaluation pipeline.

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges in subscription-based businesses. This project builds a binary classification model to identify customers who are likely to churn, enabling proactive retention strategies.

---

## 📂 Dataset

- **Source**: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: ~7,043 rows × 21 columns
- **Target Variable**: `Churn` (Yes / No)
- **Features**: Demographics, account info, and services subscribed (e.g., tenure, MonthlyCharges, Contract type, InternetService, etc.)

---

## 🧠 Models Used

| Model | Notes |
|---|---|
| XGBoost Classifier | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |
| Random Forest Classifier | `n_estimators=300` |

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgrey)

- **Language**: Python 3
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`
- **Environment**: Kaggle Notebooks / Jupyter

---

## 🔄 Project Pipeline

```
Raw CSV Data
    │
    ▼
Data Cleaning
(Drop customerID, fix TotalCharges whitespace → float)
    │
    ▼
Feature Engineering
(One-Hot Encoding for categorical columns, bool → int)
    │
    ▼
Train-Test Split (80/20, stratified)
    │
    ▼
Feature Scaling (StandardScaler)
    │
    ▼
Model Training (XGBoost, Random Forest)
    │
    ▼
Evaluation (Accuracy, ROC-AUC)
```

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| XGBoost | ~80% | ~84% |
| Random Forest | ~79% | ~82% |

> *Results may vary slightly based on environment.*

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder.

### 4. Update the file path in the notebook
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # local path
```

### 5. Run the notebook
```bash
jupyter notebook customer-churn-prediction.ipynb
```

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── customer-churn-prediction.ipynb   # Main notebook
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` or `Optuna`
- [ ] Handle class imbalance using SMOTE or `class_weight='balanced'`
- [ ] Add SHAP values for model interpretability
- [ ] Deploy as a REST API using FastAPI or Streamlit
- [ ] Add cross-validation for more robust evaluation

---

## 👤 Author

**Soham** — B.E. Computer Science, LJIET (2023–2027)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/<your-profile>)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/<your-username>)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
