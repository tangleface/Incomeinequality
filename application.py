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

# Relevant features for user input with brief descriptions
relevant_features = {
    "age": ("numeric", "Enter the age of the individual. Example: 34.01"),
    "stock_status": ("numeric", "Enter stock status. Example: 0.01"),
    "mig_year": ("numeric", "Enter the migration year. Example: 200.00"),
    "country_of_birth_father": ("category", "Select the country of birth of the father. Example: USA"),
    "employment_stat": ("category", "Select employment status (Employed/Unemployed). Example: Unemployed"),
    "tax_status": ("category", "Select tax status (Single/Married). Example: Single"),
    "citizenship": ("category", "Select citizenship status (Citizen/Non-Citizen). Example: Citizen"),
    "gender": ("category", "Select gender (Male/Female). Example: Male"),
    "industry_code": ("numeric", "Enter the industry code. Example: 0.24"),
    "working_week_per_year": ("numeric", "Enter the number of working weeks per year. Example: 40.00"),
    "wage_per_hour": ("numeric", "Enter the wage per hour. Example: 49.99"),
    "total_employed": ("numeric", "Enter the total number of employed individuals. Example: 0.00"),
    "vet_benefit": ("numeric", "Enter veteran benefits. Example: 0.00"),
    "gains": ("numeric", "Enter any financial gains. Example: 0.01"),
    "losses": ("numeric", "Enter any financial losses. Example: -0.02"),
    "education": ("category", "Select education level (High School/Bachelor/Master/PhD). Example: PhD"),
    "marital_status": ("category", "Select marital status (Single/Married/Divorced). Example: Married"),
    "race": ("category", "Select race (White/Black/Asian/Other). Example: White"),
    "household_summary": ("category", "Select household status (Living Alone/With Family/Other). Example: With Family"),
    "employment_commitment": ("category", "Select employment commitment (Full-time/Part-time/Unemployed). Example: Full-time")
}

# Encoding dictionaries (Example)
encoding_dict = {
    "country_of_birth_father": {"USA": 0, "Canada": 1, "Other": 2},
    "employment_stat": {"Employed": 0, "Unemployed": 1},
    "tax_status": {"Single": 0, "Married": 1},
    "citizenship": {"Citizen": 0, "Non-Citizen": 1},
    "gender": {"Male": 0, "Female": 1},
    "education": {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3},
    "marital_status": {"Single": 0, "Married": 1, "Divorced": 2},
    "race": {"White": 0, "Black": 1, "Asian": 2, "Other": 3},
    "household_summary": {"Living Alone": 0, "With Family": 1, "Other": 2},
    "employment_commitment": {"Full-time": 0, "Part-time": 1, "Unemployed": 2}
}

# Sidebar for user input with descriptions
st.sidebar.header("User Input Features")
user_input = {}

for feature, (dtype, description) in relevant_features.items():
    if any(feature in col for col in feature_columns):  # Ensure feature matches encoded version
        st.sidebar.write(f"**{feature}:** {description}")
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
# Explanation of Model and Feature Selection
st.subheader("Model Training and Feature Selection")
st.write("""
The model used for this prediction is a Random Forest Classifier, which is an ensemble method that builds multiple decision trees during training and outputs the class that is the mode of the classes (classification) of the individual trees. The advantage of Random Forest is that it reduces overfitting and provides better accuracy when compared to a single decision tree.

**Feature Selection**:
- The features used in the model are selected based on their correlation with income levels, domain knowledge, and their ability to improve the model's performance.
- Categorical variables like `gender`, `education`, and `employment_stat` are encoded into numeric values to allow the model to process them efficiently.
- Numerical features such as `age`, `wage_per_hour`, and `gains` directly impact the income prediction.

The model was trained on a dataset that captures various socioeconomic factors and their influence on income, and it was tuned to optimize performance using hyperparameters such as the number of trees and tree depth.
""")
# Prediction button
if st.sidebar.button("Predict"):
    y_pred = model.predict(input_df)
    st.write(f"### Predicted Income Category: {'Above Limit' if y_pred[0] == 1 else 'Below Limit'}")



