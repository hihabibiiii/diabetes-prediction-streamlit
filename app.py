import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")

st.write("Enter the patient details:")

# Inputs (PIMA diabetes dataset style)
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
bp = st.number_input("Blood Pressure", 0, 200)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ The person is NOT diabetic")
    else:
        st.error("⚠️ The person IS diabetic")
