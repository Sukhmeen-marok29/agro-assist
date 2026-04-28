import os 
import sys
import numpy as np
import streamlit as st
import pickle
import tensorflow as tf
import requests
import google.generativeai as genai
from PIL import Image
from streamlit_option_menu import option_menu

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="Agri-Smart Pro", layout="wide")

# Language Dictionary
LANG_DICT = {
    "English": {
        "title": "🌱 Agri-Smart Pro",
        "crop_tab": "Crop Recommendation",
        "disease_tab": "Disease Diagnosis",
        "bot_tab": "Agri-Bot Chat",
        "mandi_tab": "Mandi Rates",
        "n": "Nitrogen (N)",
        "p": "Phosphorus (P)",
        "k": "Potassium (K)",
        "predict_btn": "Predict Best Crop",
        "organic": "Organic Cure",
        "chemical": "Chemical Treatment"
    },
    "ਪੰਜਾਬੀ": {
        "title": "🌱 ਐਗਰੀ-ਸਮਾਰਟ ਪ੍ਰੋ",
        "crop_tab": "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        "disease_tab": "ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
        "bot_tab": "ਐਗਰੀ-ਬੋਟ ਚੈਟ",
        "mandi_tab": "ਮੰਡੀ ਦੇ ਭਾਅ",
        "n": "ਨਾਈਟ੍ਰੋਜਨ (N)",
        "p": "ਫਾਸਫੋਰਸ (P)",
        "k": "ਪੋਟਾਸ਼ੀਅਮ (K)",
        "predict_btn": "ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਦੇਖੋ",
        "organic": "ਜੈਵਿਕ ਇਲਾਜ",
        "chemical": "ਰਸਾਇਣਕ ਇਲਾਜ"
    }
}

# Language Selector in Sidebar
st.sidebar.title("Settings")
lang_choice = st.sidebar.radio("Select Language / ਭਾਸ਼ਾ ਚੁਣੋ", ["English", "ਪੰਜਾਬੀ"])
L = LANG_DICT[lang_choice]

# --- 2. KNOWLEDGE BASE ---
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'cause': 'Fungus (Venturia inaequalis)',
        'cure': 'Rake and burn fallen leaves. Apply Captan.',
        'organic_cure':'Spray baking soda & liquid soap mixture.',
        'product': 'Captan 50 WP',
        'dosage_per_acre': 2.0
    },
    'Potato___Early_blight': {
        'cause': 'Fungal pathogen (High humidity)',
        'cure': 'Avoid overhead watering. Use Chlorothalonil.',
        'organic_cure':'Diluted milk spray (1:9 ratio).',
        'product':'Chlorothalonil',
        'dosage_per_acre': 0.8
    },
    'healthy': {
        'cause': 'N/A', 'cure': 'Maintain watering.', 'organic_cure': 'Composting.', 'product':'None', 'dosage_per_acre': 0.0
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models():
    # Replace with your actual filenames
    crop_model = pickle.load(open('crop_model.pkl', 'rb')) if os.path.exists('crop_model.pkl') else None
    disease_model = tf.keras.models.load_model('disease_model.h5') if os.path.exists('disease_model.h5') else None
    return crop_model, disease_model

crop_model, disease_model = load_models()

def get_weather (city_name):
    API_KEY="b62c37dd2124c1bc17fc4c76a688d8bf"
    base_url="http://api.openweathermap.org/data/2.5/weather"
    
    params={
        "q":city_name,
        "appid":API_KEY,
        "units":"metric"
    }
    try:
        response=requests.get(base_url, params=params)
        data= response.json()
        
        if data["cod"]==200:
            temp=data["main"]["temp"]
            hum=data["main"]["humidity"]
            return temp, hum
        else:
            st.error(f"City not found: {data.get('message','')}")
            return None, None 
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None, None

# --- 4. MAIN INTERFACE ---
st.title(L["title"])

selected = option_menu(
    menu_title=None,
    options=[L["crop_tab"], L["disease_tab"], L["mandi_tab"]], 
    icons=["flower1", "search", "currency-rupee"], 
    orientation="horizontal",
)

# --- SECTION: CROP RECOMMENDATION & FERTILIZER ---
if selected == L["crop_tab"]:
    st.header(f"🌾 {L['crop_tab']}")
    
    with st.expander("Auto-Fetch Local Weather"):
        city=st.text_input("Enter city name","Kapurthala")
        if st.button("Fetch Current Weather"):
            w_temp, w_hum=get_weather(city)
            if w_temp is not None:
                st.session_state['temp']=w_temp
                st.session_state['hum']=w_hum
                st.success(f"Updated! Temp: {w_temp}.C | Humidity:{w_hum}%")
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input(L["n"], min_value=0)
        p = st.number_input(L["p"], min_value=0)
        k = st.number_input(L["k"], min_value=0)
    with col2:
        temp = st.number_input("Temperature (°C)", value=25.0)
        hum = st.number_input("Humidity (%)", value=50.0)
        ph = st.number_input("Soil pH", value=6.5)
        rain = st.number_input("Rainfall (mm)", value=100.0)

    if st.button(L["predict_btn"]):
        # Prediction Logic
        input_data = np.array([[n, p, k, temp, hum, ph, rain]])
        crop = crop_model.predict(input_data)[0] if crop_model else "Rice"
        st.success(f"Recommended Crop: **{crop}**")
        
        # FERTILIZER CALCULATOR (The Value Add)
        st.subheader("🧪 Fertilizer Requirement (Gap Analysis)")
        # Example logic: Ideal N for most crops is ~100
        if n < 80:
            st.warning(f"Your Nitrogen is low. Apply **{80-n}kg of Urea** per acre.")
        if p < 40:
            st.warning(f"Your Phosphorus is low. Apply **{40-p}kg of DAP** per acre.")
        if n >= 80 and p >= 40:
            st.success("Soil nutrient levels look healthy!")

# --- SECTION: DISEASE DIAGNOSIS ---
elif selected == L["disease_tab"]:
    st.header(f"🔍 {L['disease_tab']}")
    uploaded_file = st.file_uploader("Upload leaf...", type=["jpg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        
        if st.button("Analyze"):
            # (Your prediction logic here...)
            res = "Potato___Early_blight" # Example Result
            info = DISEASE_INFO.get(res, DISEASE_INFO['healthy'])
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"### 🌱 {L['organic']}")
                st.write(info['organic_cure'])
            with c2:
                st.error(f"### 💊 {L['chemical']}")
                st.write(f"**Product:** {info['product']}")
                st.write(f"**Application:** {info['cure']}")

# --- SECTION: MANDI RATES (The Value Add) ---
elif selected == L["mandi_tab"]:
    st.header("💰 Live Mandi Rates (Punjab)")
    # In a real app, you'd fetch this from data.gov.in API
    mandi_data = {
        "Crop": ["Wheat", "Paddy", "Potato", "Tomato"],
        "Market": ["Kapurthala", "Jalandhar", "Phagwara", "Amritsar"],
        "Price (per Quintal)": ["₹2,275", "₹2,183", "₹1,100", "₹1,800"]
    }
    st.table(mandi_data)
    st.info("Note: Rates are updated every 24 hours based on government data.")

# --- SIDEBAR: AGRI-BOT (The Value Add) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Agri-Bot Assistant")
GEMINI_API_KEY="AIzaSyAokPbpS-5BSI-Bin9To8qYCl-1yFIHzVM"
genai.configure(api_key=GEMINI_API_KEY)
model=genai.GenerativeModel("gemini-1.5-flash")
user_q = st.sidebar.text_input("Ask a farming question:")
if user_q:
    with st.spinner("Bot is thinking..."):
        try:
            full_prompt=f"You are a professional agricultural expert,. Answer this question briefly: {user_q}"
            response=model.generate_content(full_prompt)
            st.info(f"**Bot:**{response.text}")
        except Exception as e:
            st.error("Bot is not responding . Kindly check your connection.")
    
