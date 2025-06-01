import streamlit as st

# --- Your original code below, exactly as is ---
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn PRediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# --- UI upgrade starts here ---

st.markdown(f"### 🔥 গ্রাহকের প্রস্থান ঝুঁকি (Churn Probability): **{prediction_proba:.2f}** 🔥")


if prediction_proba > 0.5:
    st.markdown("""
    <div style='
        background-color:#e74c3c; 
        color:white; 
        padding:15px; 
        border-radius:8px; 
        font-size:20px; 
        font-weight:bold; 
        text-align:center; 
        box-shadow: 0 0 15px #e74c3c;'>
        ব্যাঙ্কের লগে ব্রেকআপ! বলতেছে – 'তোর সার্ভিস আর ভাল্লাগে না রে!' 💔🏦
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='
        background-color:#27ae60; 
        color:white; 
        padding:15px; 
        border-radius:8px; 
        font-size:20px; 
        font-weight:bold; 
        text-align:center; 
        box-shadow: 0 0 15px #27ae60;'>
        ব্যাঙ্ক + কাস্টোমার = জিন্দেগি লগে প্রেমের গল্প চলছে! 😍📈
    </div>
    """, unsafe_allow_html=True)

