import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Streamlit App Code
st.set_page_config(page_title="Income Prediction", layout="wide")
st.title("Income Limit Prediction App")

import os
import requests
@st.cache_resource
MODEL_URL = "https://github.com/tangleface/Incomeinequality/releases/download/v1.0/income_model.joblib"
MODEL_PATH = "income_model.joblib"

def load_model():
    # Check if the model file exists, otherwise download it
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    # Load the model
    model = joblib.load(MODEL_PATH)
    features = joblib.load("C:/Users/nabil/OneDrive/Documents/feature_columns.joblib")
    return model,features


model, feature_columns = load_model()

# Sidebar for user input
st.sidebar.header("User Input Features")

def get_user_input():
    input_data = {}
    
    # Numerical features
    input_data['age'] = st.sidebar.slider('Age', 18, 90, 30)
    input_data['hours_per_week'] = st.sidebar.slider('Weekly Hours', 10, 100, 40)
    
    # Categorical features (example)
    input_data['workclass'] = st.sidebar.selectbox('Work Class', 
                                                 ['Private', 'Self-emp', 'Government'])
    input_data['marital_status'] = st.sidebar.selectbox('Marital Status', 
                                                      ['Married', 'Single', 'Divorced'])
    
    # Add more features as needed
    return pd.DataFrame([input_data])

# Preprocess input
def preprocess_input(input_df):
    # Generate all possible dummy columns
    input_df = pd.get_dummies(input_df)
    
    # Align with training features
    final_df = pd.DataFrame(columns=feature_columns)
    
    # Fill matching columns
    for col in input_df.columns:
        if col in final_df.columns:
            final_df[col] = input_df[col]
    
    # Fill missing columns with 0
    final_df = final_df.fillna(0)
    return final_df

# Main interface
user_input = get_user_input()
processed_input = preprocess_input(user_input)

# Make prediction
threshold = st.sidebar.slider("Prediction Threshold", 0.2, 0.8, 0.5, 0.05)
prob = model.predict_proba(processed_input)[0][1]
prediction = 1 if prob >= threshold else 0

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    result = "Above Limit" if prediction == 1 else "Below Limit"
    st.success(f"**{result}** (Confidence: {prob:.2%})")
    
    st.markdown("### Model Information")
    st.write(f"Best Parameters: {model.named_steps['clf'].get_params()}")
    st.write(f"Current Threshold: {threshold:.2f}")

with col2:
    st.subheader("Model Performance")
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)
    
    # Feature Importance
    st.markdown("### Top 10 Important Features")
    importances = model.named_steps['clf'].feature_importances_
    feat_importance = pd.Series(importances, index=feature_columns)
    top_feats = feat_importance.sort_values(ascending=False).head(10)
    st.bar_chart(top_feats)

# Data summary
st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
st.sidebar.write(f"Training Samples: {len(X_train):,}")
st.sidebar.write(f"Test Samples: {len(X_test):,}")
st.sidebar.write(f"Positive Class Ratio: {y.mean():.2%}")
