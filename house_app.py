# house_app.py

import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load("house_model.pkl")
housing = fetch_california_housing(as_frame=True)
feature_names = list(housing.feature_names)

st.title("🏠 House Price Prediction App")
st.write("Enter the house details below to predict the price (in $100,000s).")

# Input form for features
inputs = []
for feature in feature_names:
    val = st.number_input(f"Enter {feature}", value=float(housing.frame[feature].mean()))
    inputs.append(val)

if st.button("Predict"):
    features = np.array([inputs])
    prediction = model.predict(features)[0]
    st.success(f"🏡 Predicted House Price: **${prediction*100000:.2f}**")
