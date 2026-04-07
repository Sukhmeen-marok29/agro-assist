import streamlit as st 
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
crop_model=pickle.load(open('crop_model.pkl','rb'))
tf.keras.models.load_model('disease_model.h5')

st.title("AGRI SMART: Intelligent Farming Assistant")
tab1, tab2=st.tabs(["Crop Recommendation", "Disease Diagnosis"])

with tab1:
    st.header("Predict best crop")
    
    n=st.number_input("Nitrogen",0,140)
    p=st.number_input("Phosphorus",0,140)
    k=st.number_input("Potassium",0,140)
    temp=st.number_input("Temperature",0,50)
    hum=st.number_input("Humidity",0,100)
    ph=st.number_input("pH",0,14)
    rain=st.number_input("Rainfall",0,300)
    
    if st.button("Recommend"):
        features=np.array([[n,p,k,temp,hum,ph,rain]])
        prediction=crop_model.predict(features)
        st.success(f"The best crop for your soil is: {prediction[0]}")
        with tab2:
            st.header("Detect leaf disease")
            uploaded_file= st.file_uploader("Upload leaf image...",type=["jpg","png"])
            if uploaded_file:
                img=Image.open(uploaded_file)
                st.image(img, width=300)
                st.write("Prediction: Tomato Bacterial Spot (Example)")
                