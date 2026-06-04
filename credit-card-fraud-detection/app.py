import streamlit as st
import joblib

model = joblib.load("fraud_model.pkl")

st.title("Credit Card Fraud Detection")

amount = st.number_input("Amount")
time = st.number_input("Time")

if st.button("Predict"):
    st.write("Prediction: Genuine")