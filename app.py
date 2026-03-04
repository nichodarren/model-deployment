import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load preprocess and model
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Heart Disease Prediction Model Deployment')

    st.subheader("Input Patient Data")

    age = st.slider('Age', 20, 100, 50)
    sex = st.slider('Sex (0 = Female, 1 = Male)', 0, 1, 1)
    cp = st.slider('Chest Pain Type (0–3)', 0, 3, 1)
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 210, 120)
    chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.slider('Fasting Blood Sugar > 120 mg/dl (0/1)', 0, 1, 0)
    restecg = st.slider('Rest ECG (0–2)', 0, 2, 1)
    thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.slider('Exercise Induced Angina (0/1)', 0, 1, 0)
    oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.0, 1.0, step=0.1)
    slope = st.slider('Slope (0–2)', 0, 2, 1)
    ca = st.slider('Number of Major Vessels (0–4)', 0, 4, 0)
    thal = st.slider('Thal (0–3)', 0, 3, 2)

    if st.button('Make Prediction'):
        features = [
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]
        result = make_prediction(features)

        if result == 1:
            st.error("⚠️ Patient has Heart Disease")
        else:
            st.success("✅ Patient does NOT have Heart Disease")


def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]


if __name__ == '__main__':
    main()
