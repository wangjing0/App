import streamlit as st
import pickle
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def load_model():
    gs_model1= joblib.load('./models/gs_model_step1_xgbc.pkl') 
    gs_model2= joblib.load('./models/gs_model_step2_xgbr.pkl') 
    X_mean = pd.read_pickle('./models/X_mean.pkl')
    return gs_model1, gs_model2, X_mean

gs_model1, gs_model2, X = load_model()

# model was trained on xgboost version==1.5.1,  version conflict
xgbr_model = gs_model2.best_estimator_['xgbr']
new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 'early_stopping_rounds', 
             'max_cat_to_onehot', 'max_leaves', 'sampling_method']

for attr in new_attrs:
    setattr(xgbr_model, attr, None)

model_calorie = Pipeline([('transformer',gs_model2.best_estimator_['transformer']),
                          ('xgbr', xgbr_model) ])

features = ['sport', 'gender', 'heart_rate_mean', 'heart_rate_std', 'altitude_mean',
           'altitude_std', 'ascend_m', 'descend_m', 'distance_total_m',
           'speed_mean', 'speed_std', 'duration_s', 'start_time']

def show_predict_page():
    st.title("🏃 🚴 🏇 🏂 🏌️ 🏄 🚣 🏊 ⛹️  🏋️ 🤸 🤼 🤽 ")
    st.write("### Today is "+ datetime.now(timezone.PST).date().isoformat() + ' '+ datetime.today().strftime('%A'))
    
    sports = (
         'bike',
         'run',
         'mountain bike',
         'bike (transport)',
         'indoor cycling',
         'walk',
         'orienteering',
         'cross-country skiing',
         'core stability training',
         'fitness walking',
         'skate',
         'roller skiing',
         'hiking',
         'circuit training',
         'kayaking',
         'rowing',
         'weight training',
         'soccer',
         'downhill skiing',
         'gymnastics'
            )
    
    genders=('male', 'female', 'unknown')

    sport = st.selectbox("Sport type", sports)
    gender = st.selectbox("Gender", genders)
    
    heart_rate = st.slider("Target Heart Rate (bpm)", 50, 220, 100)
    distance_total_km = st.slider("Total Distance (km)", 1, 42, 10)
    #speed = st.slider('Target Speed (km/h)', 1, 30, 10)
    duration_min = st.slider('Duration (min)', 5, 300, 45)
    speed = distance_total_km/duration_min/0.6 # m/s
    start_time = datetime.now().date() # time object
    
    ok = st.button("Calculate calories")
    if ok:
        X['sport'] = sport
        X['gender'] = gender
        X['heart_rate_mean'] = heart_rate
        X['speed_mean'] = speed
        X['distance_total_m'] = 1000*distance_total_km
        X['duration_s'] = 60*duration_min
        X['start_time'] = start_time
        # the feature order matters in xgboost==1.6.1!!
        y = model_calorie.predict(X[features]) 
        st.subheader(f"The burnt calorie estimation is * {np.int(y)} * cal. Good job!")
        st.subheader("燃烧吧，卡路里！！！")
        
    st.title("🏃 🚴 🏇 🏂 🏌️ 🏄 🚣 🏊 ⛹️  🏋️ 🤸 🤼 🤽 ")