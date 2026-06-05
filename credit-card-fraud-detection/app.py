import streamlit as st
import joblib

import os
import joblib

model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = joblib.load(model_path)

st.title("Credit Card Fraud Detection")

amount = st.number_input("Amount")
time = st.number_input("Time")

if st.button("Predict"):
    st.write("Prediction: Genuine")