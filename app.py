import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Heart Disease Prediction')

    st.subheader("Patient Information")

    # ===== NUMERICAL =====
    age = st.slider('Age', 20, 100, 50)
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
    thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
    oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.0, 1.0, step=0.1)

    # ===== CATEGORICAL =====
    sex = st.radio('Sex', ['Female', 'Male'])
    cp = st.selectbox('Chest Pain Type', 
                      ['Typical Angina (0)',
                       'Atypical Angina (1)',
                       'Non-anginal Pain (2)',
                       'Asymptomatic (3)'])

    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No (0)', 'Yes (1)'])

    restecg = st.selectbox('Rest ECG',
                           ['Normal (0)',
                            'ST-T abnormality (1)',
                            'Left ventricular hypertrophy (2)'])

    exang = st.radio('Exercise Induced Angina',
                     ['No (0)', 'Yes (1)'])

    slope = st.selectbox('Slope',
                         ['Upsloping (0)',
                          'Flat (1)',
                          'Downsloping (2)'])

    ca = st.selectbox('Number of Major Vessels',
                      [0,1,2,3,4])

    thal = st.selectbox('Thal',
                        ['Normal (1)',
                         'Fixed defect (2)',
                         'Reversible defect (3)'])

    if st.button('Make Prediction'):
        # Encoding manual supaya sesuai model training
        sex = 1 if sex == 'Male' else 0
        cp = int(cp[-2])
        fbs = 1 if 'Yes' in fbs else 0
        restecg = int(restecg[-2])
        exang = 1 if 'Yes' in exang else 0
        slope = int(slope[-2])
        thal = int(thal[-2])

        features = [
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]

        result = make_prediction(features)

        if result == 1:
            st.error("Patient has Heart Disease")
        else:
            st.success("Patient does NOT have Heart Disease")


def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]


if __name__ == '__main__':
    main()
