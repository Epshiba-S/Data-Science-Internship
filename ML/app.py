import streamlit as st
import pandas as pd
import pickle

# Load the brain
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Gender", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)

sex_num = 0 if sex == "male" else 1

if st.button("Predict"):
    # Predict using Pclass, Sex, Age, SibSp, Parch, Fare
    prediction = model.predict([[pclass, sex_num, age, 0, 0, fare]])
    if prediction[0] == 1:
        st.success("Survived! ")
    else:
        st.error("Did not survive. ")