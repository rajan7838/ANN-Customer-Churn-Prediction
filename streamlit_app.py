import streamlit as st
import requests
import json


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details to predict whether they will churn.")


col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)


if st.button("Predict Churn"):
    
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }

    
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        result = response.json()

        prob = result["probability"]
        pred = result["prediction"]

        
        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f" Customer will churn (probability: {prob:.2f})")
        else:
            st.success(f" Customer will NOT churn (probability: {prob:.2f})")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")