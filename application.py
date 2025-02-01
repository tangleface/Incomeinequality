import streamlit as st
import joblib
import pandas as pd
import os
import requests

# Streamlit page config
st.set_page_config(page_title="Income Prediction", layout="wide")
st.title("Income Limit Prediction App")

# Define model URL & path
MODEL_URL = "https://github.com/tangleface/Incomeinequality/releases/download/v1.0/income_model.joblib"
MODEL_PATH = "income_model.joblib"
FEATURES_PATH = "feature_columns.joblib"

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = joblib.load(MODEL_PATH)

    # Load feature names
    if os.path.exists(FEATURES_PATH):
        feature_columns = joblib.load(FEATURES_PATH)
    else:
        feature_columns = []  # Default empty if file missing

    return model, feature_columns

# Load trained model & features
model, feature_columns = load_model()

# Display feature columns for debugging
st.write("Feature columns used by the model:", feature_columns)

# Relevant features for user input
relevant_features = {
    "age": "numeric",
    "stock_status": "numeric",
    "mig_year": "numeric",
    "country_of_birth_father": "category",
    "employment_stat": "category",
    "tax_status": "category",
    "citizenship": "category",
    "gender": "category",
    "industry_code": "numeric",
    "working_week_per_year": "numeric"
}

# Encoding dictionaries (Example)
encoding_dict = {
    "country_of_birth_father": {"USA": 0, "Canada": 1, "Other": 2},
    "employment_stat": {"Employed": 0, "Unemployed": 1},
    "tax_status": {"Single": 0, "Married": 1},
    "citizenship": {"Citizen": 0, "Non-Citizen": 1},
    "gender": {"Male": 0, "Female": 1}
}

# Sidebar for user input
st.sidebar.header("User Input Features")
user_input = {}

for feature, dtype in relevant_features.items():
    if any(feature in col for col in feature_columns):  # Ensure feature matches encoded version
        if dtype == "numeric":
            user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)
        elif dtype == "category":
            options = list(encoding_dict.get(feature, {}).keys())
            selected_option = st.sidebar.selectbox(f"Select {feature}", options)
            user_input[feature] = encoding_dict.get(feature, {}).get(selected_option, -1)  # Default to -1 for unknown

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure all model features exist in input
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Fill missing encoded features with 0

# Keep only columns used by the model
input_df = input_df[feature_columns]

# Prediction button
if st.sidebar.button("Predict"):
    y_pred = model.predict(input_df)
    st.write(f"### Predicted Income Category: {'Above Limit' if y_pred[0] == 1 else 'Below Limit'}")
