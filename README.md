# Predictive Maintenance – Machine Failure Risk Prediction

## Overview
Machine learning system for predicting machine failure risk using sensor and operational data, covering EDA, preprocessing, feature engineering, model optimization, evaluation, and deployment.

## Dataset
**AI4I 2020 Predictive Maintenance Dataset (Kaggle)**  
- 10,000 machine records
- Target: `Target`
- Failure information: `Failure Type`

## Key Highlights
- Removed `UDI`, `Product ID`, and `Failure Type` to prevent data leakage.
- Engineered `Temperature_Difference`, `Power_Proxy`, and `ToolWear_Torque`.
- Used stratified splitting and threshold optimization for failure detection.
- Compared classification models and developed a regularized **Gradient Boosting Classifier**.
- Tuned the prediction threshold from `0.50` to `0.40`.
- Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- Analyzed feature importance and deployed the model using **Joblib + Streamlit**.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | **99.4%** |
| Precision | **100%** |
| Recall | **80.9%** |
| F1-Score | **89.4%** |
| ROC-AUC | **96.6%** |
| PR-AUC | **89.5%** |

## Feature Importance
- `Power_Proxy` – 34.5%
- `Temperature_Difference` – 28.3%
- `ToolWear_Torque` – 27.5%

## Technologies
**Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Joblib | Streamlit | Google Colab**
