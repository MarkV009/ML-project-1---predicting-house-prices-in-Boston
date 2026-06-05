import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Boston Housing Price Prediction")
st.write("Enter the housing features to predict median value (MEDV).")

# Input fields for all predictors
crim = st.number_input("Per capita crime rate (crim)", min_value=0.0,  format="%.5f",
    step=0.00001)
zn = st.number_input("Residential land zoned (zn)", min_value=0.0)
indus = st.number_input("Non-retail business acres (indus)", min_value=0.0)
chas = st.selectbox("Charles River dummy (chas)", [0, 1])
nox = st.number_input("Nitric oxides concentration (nox)", min_value=0.0, format="%.3f",
    step=0.001)
rm = st.number_input("Average number of rooms (rm)", min_value=0.0, format="%.3f",
    step=0.001)
age = st.number_input("Proportion of old units (age)", min_value=0.0)
dis = st.number_input("Distance to employment centers (dis)", min_value=0.0, format="%.4f",
    step=0.0001)
rad = st.number_input("Accessibility to highways (rad)", min_value=0.0)
tax = st.number_input("Property tax rate (tax)", min_value=0.0)
ptratio = st.number_input("Pupil-teacher ratio (ptratio)", min_value=0.0)
b = st.number_input("Proportion of Black residents (b)", min_value=0.0)
lstat = st.number_input("Lower status population % (lstat)", min_value=0.0)

if st.button("Predict"):
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    prediction = model.predict(features)
    st.success(f"Predicted Median Value (MEDV): {prediction[0]:.2f}")