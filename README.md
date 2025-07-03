# 📉 Customer Churn Prediction using Machine Learning

## 👤 Author
**Nami Jain**  
Computational Modeling and Data Analytics  
Virginia Tech | [GitHub Repository](https://github.com/jainnami/4824)

---

## 📌 Project Overview

This project focuses on predicting **telecommunications customer churn** using machine learning models. By analyzing demographic data, service usage patterns, contract types, and billing information, we developed a robust pipeline that effectively identifies at-risk customers. 

The project uses the **IBM Telco Customer Churn Dataset** and compares the performance of several supervised learning models, culminating in a tuned LightGBM model achieving high predictive accuracy.

---

## 🎯 Objectives

- Predict customer churn (binary classification: Churn vs. No Churn)
- Improve recall for churners to enable proactive retention strategies
- Compare multiple models using metrics like F1-Score and ROC-AUC
- Use feature importance for interpretability and business insight

---

## 🗂️ Dataset

- **Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Target variable:** `Churn` (`Yes` or `No`)
- **Features:** Demographics, billing, contract types, internet/phone services

---

## 🛠️ Tools & Technologies

- **Language:** Python 3
- **Libraries:** `pandas`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`, `SMOTE`
- **Techniques:** One-hot encoding, feature engineering, model tuning, SMOTE for class imbalance

---

## 🧪 Models & Performance

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.78     | 0.68      | 0.70   | 0.69     | 0.83    |
| Random Forest      | 0.81     | 0.72      | 0.76   | 0.74     | 0.86    |
| Gradient Boosting  | 0.80     | 0.83      | 0.80   | 0.81     | 0.75    |
| **LightGBM**       | **0.82** | **0.74**  | **0.78**| **0.76** | **0.87** |

✅ **LightGBM** performed the best, balancing high recall and precision while offering interpretability and scalability.

---

## 📊 Feature Importance Highlights

- **MonthlyCharges** – Most influential feature (high billing → higher churn)
- **Tenure** – Shorter tenure → more likely to churn
- **ContractType** – Month-to-month customers more likely to churn
- **TotalCharges** – Correlated with loyalty

These features provided actionable insights for telecom companies to improve retention strategies.

---

## ⚙️ Methodology Summary

1. **Preprocessing**
   - One-hot encoding for categorical variables
   - Normalization of numerical features
   - Handling missing/invalid values

2. **Modeling**
   - Split: 70% train / 15% validation / 15% test
   - Applied SMOTE to address class imbalance
   - Cross-validation and hyperparameter tuning (GridSearch)

3. **Evaluation**
   - Used Accuracy, Precision, Recall, F1, ROC-AUC
   - ROC curve analysis and confusion matrix evaluation

4. **Interpretability**
   - Feature importance charts from LightGBM & Gradient Boosting
   - Targeted business insights for contract and pricing strategy

---

## 🔍 Key Challenges

- **Class imbalance** (26% churners): addressed via SMOTE
- **Recall vs. Precision tradeoff**: resolved using F1-Score and ROC-AUC
- **Model complexity**: LightGBM offered better performance with reduced training time compared to Random Forest

---

## 📁 Repository Contents

