#to load models
import pickle
import joblib
import xgboost as xgb
import pandas as pd

#to create the streamlit api
import streamlit as st
# xgb model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")
# the treshold for the predicts was 0.1


# scalers
ss = joblib.load('standard_scaler.gz')
#knn = joblib.load('knn_imputer.gz')
ord_enc = joblib.load('ordinal_encoder.gz')

#Caching the model for faster loading
@st.cache

def predict(gender, age, hypertension, heart_disease, ever_married, work_type,
            residence_type, avg_glucose_level, bmi, smoking_status):
    """
    A function to predict the risk of stroke.

    :return: 0: no stroke, 1 : stroke
    """
    df = pd.DataFrame([[gender, age, hypertension,
                       heart_disease, ever_married,
                       work_type, residence_type, avg_glucose_level,
                       bmi, smoking_status]],
                      columns=['gender', 'age', 'hypertension', 'heart_disease',
                               'ever_married', 'work_type', 'Residence_type',
                               'avg_glucose_level', 'bmi', 'smoking_status'])
    # get columns for scaling
    numerics = ['age', 'avg_glucose_level', 'bmi']
    categories = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                  'Residence_type', 'smoking_status']

    df[categories] = pd.DataFrame(ord_enc.transform(df[categories]), columns=df[categories].columns)

    df[numerics] = pd.DataFrame(ss.transform(df[numerics]), columns=df[numerics].columns)


    # make a prediction
    pred = model.predict(df)
    prediction = [1 if i >= 0.1 else 0 for i in pred]

    return prediction

# streamlit app
st.title('Stroke Prediction')
st.header('Enter your information:')


gender = st.selectbox(label='Gender:',
                     options=['Female', 'Male', 'Other'])
age = st.number_input('Age:', min_value=0.01, max_value=120.0, value=1.0)

hypertension = st.selectbox(label='Hypertension:',
                            options=['No', 'Yes'])
heart_disease = st.selectbox(label='Heath Disease:',
                             options=['No', 'Yes'])
ever_married = st.selectbox(label='Ever Married:',
                            options=['No', 'Yes'])
work_type = st.selectbox(label='Work Type:',
                         options=['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
residence_type = st.selectbox(label='Residency:',
                              options=['Rural', 'Urban'])
avg_glucose_level = st.number_input('Average Blood Glucose Level:',
                                    min_value=0.1,
                                    max_value=1500.0,
                                    value=1.0)
height = st.number_input('Your height in meters',
                         min_value=0.01,
                         max_value=3.00,
                         value=0.1)
weight = st.number_input('Your body weight in kg',
                         min_value=0.1,
                         max_value=500.0,
                         value=1.0)
bmi = weight / pow(height, 2)

smoking_status = st.selectbox('Smoking:',
                              options=['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

button = st.button('Click to predict')

if button:

    outcome = predict(gender, age, hypertension, heart_disease, ever_married, work_type,
                      residence_type, avg_glucose_level, bmi, smoking_status)

    if outcome[0] == 1:
        st.warning('Urgently consult your physician')
    else:
        st.success('You seem good, keep up looking good to yourself.')




