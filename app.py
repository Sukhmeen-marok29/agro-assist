import os 
import sys
import numpy as np
import streamlit as st
import pickle
import tensorflow as tf
from PIL import Image
from streamlit_option_menu import option_menu  # New Import

# THE BRIDGE: Prevents 'numpy._core' errors
try:
    import numpy.core.multiarray
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError:
    pass    

# 1. KNOWLEDGE BASE
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'cause': 'Fungus (Venturia inaequalis) that overwinters in fallen leaves.',
        'cure': 'Rake and burn fallen leaves. Apply fungicides like Captan during budding.',
        'product': 'Captan 50 WP',
        'dosage_per_acre':2.0
    },
    'Corn___Common_rust': {
        'cause': 'Fungus (Puccinia sorghi) spread by wind.',
        'cure': 'Plant resistant hybrids. Apply fungicides if infection is severe.',
        'product':'Pyraclostrobin',
        'dosage_per_acre':0.15
    },
    'Potato___Early_blight': {
        'cause': 'Fungal pathogen thriving in high humidity.',
        'cure': 'Avoid overhead watering. Apply chlorothalonil-based fungicides.',
        'product':'Chlorothalonil',
        'dosage_per_acre':0.8
    },
    'Potato___Late_blight': {
        'cause': 'Water mold (Phytophthora infestans); spreads in cool, wet weather.',
        'cure': 'Remove infected plants immediately. Use copper-based fungicides.',
        'product':'Copper Oxychloride',
        'dosage_per_acre':1.5
    },
    'healthy': {
        'cause': 'N/A',
        'cure': 'Keep up the good work! Maintain regular watering.',
        'product':'None',
        'dosage_per_acre':0.0
        
    }
}

# 2. LOAD MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.h5')

@st.cache_resource
def load_models():
    loaded_crop_model = None
    loaded_disease_model = None
    if os.path.exists(CROP_MODEL_PATH):
        try:  
            with open(CROP_MODEL_PATH, 'rb') as f:
                loaded_crop_model = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading Crop Model: {e}")
    
    if os.path.exists(DISEASE_MODEL_PATH):
        try:
            loaded_disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading Disease Model: {e}")
             
    return loaded_crop_model, loaded_disease_model

crop_model, disease_model = load_models()
          
def main():
    # --- TOP NAVIGATION MENU ---
    # This replaces the sidebar selectbox
    selected = option_menu(
        menu_title=None, # No title for the horizontal bar
        options=["Crop Recommendation", "Disease Diagnosis"], 
        icons=["flower1", "search"], # Bootstrap icons
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal",
    )

    # --- SECTION 1: CROP RECOMMENDATION ---
    if selected == "Crop Recommendation":
        st.header("🌾 Get Crop Recommendations")
        st.markdown("Enter soil parameters to find the best crop for your land.")
        
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nitrogen (N)", min_value=0)
            p = st.number_input("Phosphorus (P)", min_value=0)
            k = st.number_input("Potassium (K)", min_value=0)
            temp = st.number_input("Temperature (°C)", format="%.2f")
        with col2:
            humidity = st.number_input("Humidity (%)", format="%.2f")
            ph = st.number_input("Soil pH level", format="%.2f")
            rainfall = st.number_input("Rainfall (mm)", format="%.2f")

        if st.button("Recommend Crop"):
            if crop_model is not None:
                input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
                prediction = crop_model.predict(input_data)
                st.success(f"The recommended crop is: **{prediction[0]}**")
            else:
                st.error("Crop model is not loaded correctly.")

    # --- SECTION 2: DISEASE DIAGNOSIS ---
    elif selected == "Disease Diagnosis":
        st.header("🔍 Identify Plant Diseases")
        uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Diagnose Disease"):
                if disease_model is not None:
                    try:
                        target_size = (200, 200) 
                        img = image.resize(target_size) 
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        prediction = disease_model.predict(img_array)
                        confidence = np.max(prediction)
                        result_index = np.argmax(prediction)
                        
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
                        
                        predicted_class = classes[result_index]
                        
                        if confidence > 0.50:
                            predicted_class=classes[result_index]
                            st.success(f"**Result:** {predicted_class} ({confidence*100:.2f}% Confidence)")
                            info = DISEASE_INFO.get(predicted_class, DISEASE_INFO.get('healthy'))
                            
                            if "healthy" not in predicted_class.lower():
                                st.markdown("---")
                                st.subheader("Treatment & Dosage Calculator")
                                
                                acres=st.number_input("Enter your land size (in Acres)", min_value=0.1, value=1.0,step=0.5)
                               
                                dosage_val=info.get('dosage_per_acre',0)
                                total_needed=acres* dosage_val
                                
                                d_col1, d_col2 = st.columns(2)
                                with d_col1:
                                    st.warning("### ⚠️ Common Causes")
                                    st.write(info['cause'])
                                    st.info("###✅ Recommentded Product")
                                    st.write(f"**{info['product']}**")
                                    
                                with d_col2:
                                    st.success("###Required Dosage")
                                    unit ="Litres" if "liquid" in info['product'].lower() or info['dosage_per_acre']<0.5 else "kg"
                                    st.metric(label=f"Total{info['product']} needed",value=f"{total_needed:.2f}{unit}")
                                    st.write(f"**Application:**{info['cure']}")
                                    st.write(f"* Tip: Mix the {total_needed:.2f}{unit} with approx. {int (acres * 200)}L of water.*")
                            else:
                                st.balloons()
                                st.success("Your plant looks healthy!")
                        else:
                            st.warning("Low confidence. Please try a clearer image.")
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()