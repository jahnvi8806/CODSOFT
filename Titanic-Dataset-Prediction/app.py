import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Preprocessing
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

features = ["Pclass", "Sex", "Age", "Fare"]
X = df[features]
y = df["Survived"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title(" Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

sex_value = 0 if sex == "male" else 1

if st.button("Predict"):
    prediction = model.predict([[pclass, sex_value, age, fare]])

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")