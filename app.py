import sys
import numpy as np

# THE BRIDGE: Prevents 'numpy._core' errors if your environment updates
try:
    import numpy.core.multiarray
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError:
    pass

import streamlit as st
import pickle
import tensorflow as tf
from PIL import Image

# 1. LOAD MODELS
try:
    # Adjust paths if your files are in a different folder
    crop_model = pickle.load(open('crop_model.pkl', 'rb'))
    disease_model = tf.keras.models.load_model('disease_model.h5', compile=False)
except Exception as e:
    st.error(f"Error loading models: {e}")

def main():
    st.title("AGRI SMART: Intelligent Farming Assistant")
    
    # Sidebar Navigation
    menu = ["Crop Recommendation", "Disease Diagnosis"]
    choice = st.sidebar.selectbox("Menu", menu)

    # --- SECTION 1: CROP RECOMMENDATION ---
    if choice == "Crop Recommendation":
        st.subheader("Get Crop Recommendations")
        
        # All inputs aligned correctly to prevent IndentationErrors
        n = st.number_input("Enter Nitrogen (N)", min_value=0)
        p = st.number_input("Enter Phosphorus (P)", min_value=0)
        k = st.number_input("Enter Potassium (K)", min_value=0)
        temp = st.number_input("Temperature (°C)", format="%.2f")
        humidity = st.number_input("Humidity (%)", format="%.2f")
        ph = st.number_input("Soil pH level", format="%.2f")
        rainfall = st.number_input("Rainfall (mm)", format="%.2f")

        if st.button("Recommend Crop"):
            input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
            prediction = crop_model.predict(input_data)
            st.success(f"The recommended crop is: {prediction[0]}")

    # --- SECTION 2: DISEASE DIAGNOSIS ---
    elif choice == "Disease Diagnosis":
        st.subheader("Identify Plant Diseases")
        uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Diagnose Disease"):
                try:
                    # Resize to 200x200 as required by your model
                    target_size = (200, 200) 
                    img = image.resize(target_size) 
                    
                    # Preprocessing
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediction
                    prediction = disease_model.predict(img_array)
                    confidence = np.max(prediction)
                    result_index = np.argmax(prediction)
                    
                    # Full PlantVillage Class List (38 classes)
                    classes = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
                        'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 
                        'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
                        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]
                    
                    # Display result with confidence threshold
                    if confidence > 0.50:
                        st.success(f"Prediction: {classes[result_index]} ({confidence*100:.2f}% Confidence)")
                    else:
                        st.warning("Low confidence. The model might not recognize this specific plant.")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()
