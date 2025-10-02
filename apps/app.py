import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# =========================
# Path Setup
# =========================
BASE_DIR = r"C:\Users\USER\Desktop\Testing\Churning"

DATA_PATH = os.path.join(BASE_DIR, "Data", "Churning_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# =========================
# Load Model + Preprocessors
# =========================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}. Please place 'best_model.pkl' in the 'models/' folder.")
        return None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Dataset not found at {DATA_PATH}. Please place 'Churning_cleaned.csv' in the Data folder.")
        return None
    return pd.read_csv(DATA_PATH)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction"])

# =========================
# Load resources
# =========================
model, encoder, scaler = load_artifacts()
data = load_data()

# =========================
# Pages
# =========================
if page == "Home":
    st.title("📊 Customer Churn Analysis App")
    st.write(
        """
        Welcome to the **Customer Churn Prediction App**.
        - Use the **EDA** section to explore the dataset.
        - Use the **Prediction** section to check churn probability for single or multiple customers.
        """
    )

elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if data is not None:
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        st.write("### Dataset Info")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        st.write("### Missing Values")
        st.write(data.isnull().sum())

else:  # Prediction Page
    st.title("🔮 Predict Customer Churn")
    if model is None or encoder is None or scaler is None:
        st.stop()

    input_method = st.radio("Choose Input Method", ["Single Customer", "Upload CSV"])

    # Default features
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Let user select which features to keep
    st.sidebar.write("⚙️ Feature Selection")
    selected_categorical = st.sidebar.multiselect(
        "Select Categorical Features", categorical_features, default=categorical_features
    )
    selected_numerical = st.sidebar.multiselect(
        "Select Numerical Features", numerical_features, default=numerical_features
    )

    # ------------------- Single Customer -------------------
    if input_method == "Single Customer":
        st.subheader("Enter details for a single customer:")

        user_input = {}
        for feature in selected_categorical:
            if feature == "gender":
                val = st.selectbox(feature, options=["Male", "Female"])
            elif feature == "InternetService":
                val = st.selectbox(feature, options=["DSL", "Fiber optic", "No"])
            elif feature == "Contract":
                val = st.selectbox(feature, options=["Month-to-month", "One year", "Two year"])
            elif feature == "PaymentMethod":
                val = st.selectbox(
                    feature,
                    options=[
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ]
                )
            else:
                val = st.selectbox(feature, options=["Yes", "No"])
            user_input[feature] = val

        for feature in selected_numerical:
            val = st.number_input(feature, min_value=0.0, step=1.0)
            user_input[feature] = val

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])

            try:
                # Apply preprocessing only on selected features
                cat_data = encoder.transform(input_df[selected_categorical]) if selected_categorical else np.empty((1, 0))
                num_data = scaler.transform(input_df[selected_numerical]) if selected_numerical else np.empty((1, 0))
                X = np.hstack((cat_data, num_data))

                prediction = model.predict(X)[0]

                probabilities = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    if proba.shape[1] > 1:
                        probabilities = proba[0, 1]
                    else:
                        probabilities = proba[0, 0]

                st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
                if probabilities is not None:
                    st.info(f"Estimated probability of churn: {probabilities:.2%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ------------------- Upload CSV -------------------
    else:
        st.subheader("Upload customer data (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.write("### Uploaded Data Preview")
                st.dataframe(upload_df.head())

                # Apply preprocessing only on selected features
                cat_data = encoder.transform(upload_df[selected_categorical]) if selected_categorical else np.empty((len(upload_df), 0))
                num_data = scaler.transform(upload_df[selected_numerical]) if selected_numerical else np.empty((len(upload_df), 0))
                X = np.hstack((cat_data, num_data))

                if st.button("Predict for Uploaded Data"):
                    predictions = model.predict(X)

                    probabilities = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)
                        if proba.shape[1] > 1:
                            probabilities = proba[:, 1]
                        else:
                            probabilities = proba[:, 0]

                    results = upload_df.copy()
                    results["Prediction"] = ["Churn" if p == 1 else "No Churn" for p in predictions]
                    if probabilities is not None:
                        results["Probability"] = probabilities

                    st.write("### Predictions")
                    st.dataframe(results.head())
            except Exception as e:
                st.error(f"Error processing file: {e}")
