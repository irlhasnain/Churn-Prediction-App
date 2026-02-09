# Gender -> 1 Female  0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# order of  the x ->'Age', 'Gender', 'Tenure', 'MonthlyCharges'


import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Predicition App")

st.divider()

st.write("Please enter the value and hit the predict button for getting a prediction")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0,max_value=130,value=10)

monthlycharges = st.number_input("Enter Monthly Charges",min_value=0, max_value=150)

gender = st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predictbutton = st.button("Predict!")

st.divider()

if predictbutton:

    gender_selected = 1 if gender == "female" else 0

    x=[age,gender_selected,tenure,monthlycharges]

    x1=np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"predicted : {predicted}")

else:
    st.write("Please enter the values and use predict button")