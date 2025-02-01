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

# Sidebar for user input
st.sidebar.header("User Input Features")

user_input = {}
for feature in feature_columns:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Prediction button
if st.sidebar.button("Predict"):
    y_pred = model.predict(input_df)
    st.write(f"### Predicted Income Category: {'Above Limit' if y_pred[0] == 1 else 'Below Limit'}")
