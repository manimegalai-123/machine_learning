import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = RandomForestClassifier(n_estimators=100)

data = pd.read_csv("HCV-Egy-Data.csv")
X = data.iloc[:, :-1]
y = data['Baselinehistological staging']
model.fit(X, y)

# Streamlit UI
st.title("HCV BaselineHistological staging Prediction")
st.write("Enter the required details to predict the grading")

# Taking user input
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Baseline Histological Grading: {prediction}")
