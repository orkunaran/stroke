#to load models
import pickle
import joblib
import xgboost as xgb

#to create the streamlit api
import streamlit as st
# xgb model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# scalers
ss = joblib.load('standard_scaler.gz')
knn = joblib.load('knn_imputer.gz')
ord_enc = joblib.load('ordinal_encoder.gz')

# streamlit app
st.write('### Stroke Prediction API')