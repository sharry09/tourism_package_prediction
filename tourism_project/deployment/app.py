import streamlit as st
import joblib
import pandas as pd

st.title("Wellness Tourism Package Prediction")

model = joblib.load("tourism_project/deployment/best_model.pkl")

age = st.number_input("Age", 18, 80)
income = st.number_input("Monthly Income", 1000, 100000)
trips = st.number_input("Number of Trips", 0, 20)
passport = st.selectbox("Passport", [0,1])
owncar = st.selectbox("Own Car", [0,1])

df = pd.DataFrame([[age,income,trips,passport,owncar]],
                  columns=["Age","MonthlyIncome","NumberOfTrips","Passport","OwnCar"])

if st.button("Predict"):
    result = model.predict(df)
    st.write("Prediction:", "Will Purchase" if result[0]==1 else "Will Not Purchase")
