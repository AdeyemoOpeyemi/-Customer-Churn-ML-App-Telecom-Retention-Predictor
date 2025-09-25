#  Customer Churn ML App — Telecom Retention Predictor

This project provides a **machine learning pipeline** and an **interactive Streamlit web app** to predict **customer churn** using telecom customer data.  
It includes **EDA (Exploratory Data Analysis)** and **prediction tools** for single customers or batch CSV uploads.

---

##  Features

- **EDA Dashboard**:
  - Data overview (head, summary statistics).
  - Distribution plots of target (`Churn`) and numerical features.
- **Prediction Dashboard**:
  - Predict churn for a **single customer** with manual inputs.
  - Upload **CSV** with customer data for batch predictions.
- **Robust CSV Handling**:
  - Automatically fills missing columns with defaults.
  - Encodes categorical variables and scales numeric features.
- **Outputs**:
  - Churn prediction (`CHURN` or `STAY`).
  - Churn probability (confidence score).
  - Downloadable CSV with predictions.

---

---

##  Dataset

- **Name**: Telecom Customer Churn dataset (cleaned as `Churning_cleaned.csv`)
- **Target Variable**: `Churn` (Yes/No → binary classification)
- **Key Features**:
  - Categorical: `gender`, `Partner`, `Dependents`, `InternetService`, `Contract`, `PaymentMethod`, etc.
  - Numerical: `tenure`, `MonthlyCharges`, `TotalCharges`
- **Processing**:
  - Categorical encoded with OneHotEncoder.
  - Numerical scaled with StandardScaler.

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/customer-churn-ml-app.git
   cd customer-churn-ml-app
