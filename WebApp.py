# -*- coding: utf-8 -*-
"""
Created on Sat July  14 21:24:53 2021

@author: aatif
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""Diabetes Detection using Machine Learning and Python""")

image = Image.open('D:\Data Science/sugar.jpeg')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv('diabetes.csv')

st.subheader('Data Information')

st.dataframe(df)

st.write(df.describe())

chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
y = df.iloc[:,-1].values

#Split data i train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature input
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    user_data = {'pregnancies': pregnancies,
             'glucose': glucose,
             'blood_pressure': blood_pressure,
             'skin_thickness': skin_thickness,
             'insulin': insulin, 
             'BMI': BMI,
             'DPF': DPF,
             'age': age
            }
    
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification: ')
st.write(prediction)
