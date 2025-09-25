# ===============================
# app.py - Customer Churn Prediction (Robust CSV Handling)
# ===============================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================
# Load artifacts
# ==================================
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Load cleaned dataset for options
df = pd.read_csv("Churning_cleaned.csv")

categorical_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in categorical_cols:
    df[col] = df[col].astype(str)

# ==================================
# Sidebar navigation
# ==================================
st.sidebar.title("Navigation")

# Enable EDA option
show_eda = st.sidebar.checkbox("Enable EDA Page", value=True)

pages = ["Home", "Prediction"]
if show_eda:
    pages.insert(1, "EDA")

page = st.sidebar.radio("Go to", pages)

# ==================================
# Home Page
# ==================================
if page == "Home":
    st.title("📊 Customer Churn Prediction App")
    st.write("""
        Welcome!  
        - Use the **EDA page** to explore data (if enabled).  
        - Use **Prediction page** to check customer churn.  
    """)

# ==================================
# EDA Page
# ==================================
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Numerical Feature Distributions")
    for col in numerical_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# ==================================
# Prediction Page
# ==================================
elif page == "Prediction":
    st.title("🔮 Predict Customer Churn")

    st.subheader("Choose Input Method")
    input_method = st.radio("Select input method:", ["Single Customer", "Upload CSV"])
    input_df = None

    # -----------------------------
    # Single customer input
    # -----------------------------
    if input_method == "Single Customer":
        st.write("Enter details for a single customer:")

        if "inputs" not in st.session_state:
            st.session_state.inputs = {}

        user_data = {}
        st.subheader("Select Features to Include")
        selected_cats = st.multiselect("Categorical Features", categorical_cols, default=categorical_cols)
        selected_nums = st.multiselect("Numerical Features", numerical_cols, default=numerical_cols)

        for col in selected_cats:
            options = df[col].unique().tolist()
            default_val = st.session_state.inputs.get(col, options[0])
            user_data[col] = st.selectbox(f"{col}", options, index=options.index(default_val))

        for col in selected_nums:
            default_val = st.session_state.inputs.get(col, 0.0)
            user_data[col] = st.number_input(f"{col}", min_value=0.0, value=float(default_val))

        st.session_state.inputs = user_data
        input_df = pd.DataFrame([user_data])

    # -----------------------------
    # Multiple customers via CSV
    # -----------------------------
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV with customer data", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(input_df.head())

            # Automatically select only features available in CSV that are in training
            available_cats = [c for c in categorical_cols if c in input_df.columns]
            available_nums = [c for c in numerical_cols if c in input_df.columns]

            st.subheader("Select Features to Include")
            selected_cats = st.multiselect("Categorical Features", available_cats, default=available_cats)
            selected_nums = st.multiselect("Numerical Features", available_nums, default=available_nums)
        else:
            st.warning("Please upload a CSV file.")

    # -----------------------------
    # Process input
    # -----------------------------
    if input_df is not None:
        # Fill missing categorical columns
        for col in selected_cats:
            if col not in input_df.columns:
                input_df[col] = "Unknown"  # placeholder for missing
            else:
                input_df[col] = input_df[col].astype(str)

        # Fill missing numerical columns
        for col in selected_nums:
            if col not in input_df.columns:
                input_df[col] = 0
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # Ensure columns are in the same order as training
        X_cat = input_df[selected_cats] if selected_cats else pd.DataFrame(np.empty((len(input_df), 0)))
        X_num = input_df[selected_nums] if selected_nums else pd.DataFrame(np.empty((len(input_df), 0)))

        # Encode categorical
        if selected_cats:
            input_encoded = encoder.transform(X_cat)
            if hasattr(input_encoded, "toarray"):
                input_encoded = input_encoded.toarray()
        else:
            input_encoded = np.empty((len(input_df), 0))

        # Scale numerical
        if selected_nums:
            input_scaled = scaler.transform(X_num)
        else:
            input_scaled = np.empty((len(input_df), 0))

        # Combine features
        final_input = np.hstack((input_encoded, input_scaled))

        # -----------------------------
        # Make prediction
        # -----------------------------
        if st.button("Predict"):
            predictions = model.predict(final_input)
            probabilities = model.predict_proba(final_input)[:, 1]

            results_df = input_df[selected_cats + selected_nums].copy()
            results_df["Churn_Prediction"] = ["CHURN" if p == 1 else "STAY" for p in predictions]
            results_df["Churn_Probability"] = probabilities.round(2)

            st.subheader("Prediction Results")
            st.dataframe(results_df)

            # Download CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        if st.button("Reset Inputs"):
            st.session_state.inputs = {}
            st.experimental_rerun()
