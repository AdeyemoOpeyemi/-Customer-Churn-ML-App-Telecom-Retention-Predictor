# ===============================
# app_console.py - Customer Churn Prediction (Console Version)
# ===============================
import pandas as pd
import joblib
import numpy as np

# ==================================
# Load artifacts
# ==================================
try:
    model = joblib.load("best_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    print(f"❌ Missing file: {e}")
    exit()

# Load cleaned dataset for options
try:
    df = pd.read_csv("Churning_cleaned.csv")
except FileNotFoundError:
    print("❌ Churning_cleaned.csv not found.")
    exit()

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
# Console UI
# ==================================
def main():
    print("📊 Customer Churn Prediction Console App")
    print("Type 'exit' anytime to quit.\n")

    while True:
        print("\nSelect input method:")
        print("1 - Single Customer")
        print("2 - Upload CSV")
        choice = input("Enter 1 or 2: ").strip()
        if choice.lower() == "exit":
            break

        input_df = None
        selected_cats = categorical_cols
        selected_nums = numerical_cols

        # -----------------------------
        # Single customer input
        # -----------------------------
        if choice == "1":
            user_data = {}
            print("\nEnter details for a single customer:")
            for col in selected_cats:
                options = df[col].unique().tolist()
                val = input(f"{col} (choose from {options[0:5]}...): ").strip()
                if val.lower() == "exit":
                    return
                user_data[col] = val if val else options[0]

            for col in selected_nums:
                val = input(f"{col} (numeric): ").strip()
                if val.lower() == "exit":
                    return
                try:
                    user_data[col] = float(val)
                except:
                    user_data[col] = 0.0

            input_df = pd.DataFrame([user_data])

        # -----------------------------
        # Multiple customers via CSV
        # -----------------------------
        elif choice == "2":
            path = input("Enter CSV file path: ").strip()
            if path.lower() == "exit":
                break
            try:
                input_df = pd.read_csv(path)
                print("Preview of uploaded data:")
                print(input_df.head())
            except FileNotFoundError:
                print("❌ File not found.")
                continue

            # Only keep features that exist in both CSV and training
            selected_cats = [c for c in categorical_cols if c in input_df.columns]
            selected_nums = [c for c in numerical_cols if c in input_df.columns]

        else:
            print("❌ Invalid choice.")
            continue

        # -----------------------------
        # Preprocess input
        # -----------------------------
        for col in selected_cats:
            if col not in input_df.columns:
                input_df[col] = "Unknown"
            else:
                input_df[col] = input_df[col].astype(str)

        for col in selected_nums:
            if col not in input_df.columns:
                input_df[col] = 0
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # Encode categorical
        if selected_cats:
            input_encoded = encoder.transform(input_df[selected_cats])
            if hasattr(input_encoded, "toarray"):
                input_encoded = input_encoded.toarray()
        else:
            input_encoded = np.empty((len(input_df), 0))

        # Scale numerical
        if selected_nums:
            input_scaled = scaler.transform(input_df[selected_nums])
        else:
            input_scaled = np.empty((len(input_df), 0))

        # Combine features
        final_input = np.hstack((input_encoded, input_scaled))

        # -----------------------------
        # Make prediction
        # -----------------------------
        predictions = model.predict(final_input)
        probabilities = model.predict_proba(final_input)[:, 1]

        results_df = input_df[selected_cats + selected_nums].copy()
        results_df["Churn_Prediction"] = ["CHURN" if p == 1 else "STAY" for p in predictions]
        results_df["Churn_Probability"] = probabilities.round(2)

        print("\n✅ Prediction Results:")
        print(results_df)

        # Save to CSV
        save = input("Do you want to save results to CSV? (y/n): ").strip().lower()
        if save == "y":
            out_file = input("Enter output CSV file name: ").strip()
            results_df.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")

        cont = input("\nDo you want to predict again? (y/n): ").strip().lower()
        if cont != "y":
            break

if __name__ == "__main__":
    main()
