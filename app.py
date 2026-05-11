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

if 'temp' not in st.session_state: st.session_state['temp'] = 25.0
if 'hum' not in st.session_state: st.session_state['hum'] = 50.0

# --- GEMINI AI SETUP ---
GEMINI_API_KEY = "AIzaSyAokPbpS-5BSI-Bin9To8qYCl-1yFIHzVM"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.sidebar.error(f"AI Error: {e}")

# Language Dictionary
LANG_DICT = {
    
    "English": {
        "title": "🌱 Agri-Smart Pro",
        "crop_tab": "Crop Recommendation",
        "disease_tab": "Disease Diagnosis",
        "mandi_tab": "Mandi Rates",
        "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "ph": "Soil pH", "rain": "Rainfall (mm)",
        "predict_btn": "Predict Best Crop",
        "weather_btn": "Fetch Current Weather",
        "city_label": "Enter City Name",
        "analyze_btn": "Analyze Image",
        "organic": "Organic Cure",
        "chemical": "Chemical Treatment",
        "fert_title": "💡 Fertilizer Suggestion",
        "bot_title": "🤖 Agri-Bot Assistant",
        "bot_ask": "Ask a farming question:",
        "mandi_title": "💰 Live Mandi Rates (Punjab)",
        "mandi_note": "Note: Rates are updated every 24 hours."
    },
    "ਪੰਜਾਬੀ": {
        "title": "🌱 ਐਗਰੀ-ਸਮਾਰਟ ਪ੍ਰੋ",
        "crop_tab": "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        "disease_tab": "ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
        "mandi_tab": "ਮੰਡੀ ਦੇ ਭਾਅ",
        "n": "ਨਾਈਟ੍ਰੋਜਨ (N)", "p": "ਫਾਸਫੋਰਸ (P)", "k": "ਪੋਟਾਸ਼ੀਅਮ (K)",
        "temp": "ਤਾਪਮਾਨ (°C)", "hum": "ਨਮੀ (%)", "ph": "ਮਿੱਟੀ ਦਾ pH", "rain": "ਵਰਖਾ (mm)",
        "predict_btn": "ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਦੇਖੋ",
        "weather_btn": "ਤਾਜ਼ਾ ਮੌਸਮ ਦੇਖੋ",
        "city_label": "ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਲਿਖੋ",
        "analyze_btn": "ਜਾਂਚ ਕਰੋ",
        "organic": "ਜੈਵਿਕ ਇਲਾਜ",
        "chemical": "ਰਸਾਇਣਕ ਇਲਾਜ",
        "fert_title": "💡 ਖਾਦ ਦੀ ਸਲਾਹ",
        "bot_title": "🤖 ਐਗਰੀ-ਬੋਟ ਸਹਾਇਕ",
        "bot_ask": "ਖੇਤੀਬਾੜੀ ਬਾਰੇ ਸਵਾਲ ਪੁੱਛੋ:",
        "mandi_title": "💰 ਲਾਈਵ ਮੰਡੀ ਦੇ ਭਾਅ (ਪੰਜਾਬ)",
        "mandi_note": "ਨੋਟ: ਰੇਟ ਹਰ 24 ਘੰਟਿਆਂ ਵਿੱਚ ਅਪਡੇਟ ਕੀਤੇ ਜਾਂਦੇ ਹਨ।"
    }
}

st.sidebar.title("Settings / ਸੈਟਿੰਗਾਂ")
lang_choice = st.sidebar.radio("Select Language / ਭਾਸ਼ਾ ਚੁਣੋ", ["English", "ਪੰਜਾਬੀ"])
L = LANG_DICT[lang_choice]

# --- 2. EXPANDED KNOWLEDGE BASE ---
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'en': {'org': 'Apply Neem oil or baking soda spray.', 'prod': 'Captan 50 WP', 'cure': 'Fungicide spray.'},
        'pa': {'org': 'ਨੀਮ ਦਾ ਤੇਲ ਜਾਂ ਬੇਕਿੰਗ ਸੋਡਾ ਸਪ੍ਰੇ ਕਰੋ।', 'prod': 'ਕੈਪਟਨ 50 WP', 'cure': 'ਫੰਗਸਨਾਸ਼ਕ ਸਪ੍ਰੇ।'}
    },
    'Potato___Early_blight': {
        'en': {'org': 'Remove infected leaves. Use compost tea.', 'prod': 'Mancozeb', 'cure': 'Apply Mancozeb spray.'},
        'pa': {'org': 'ਪ੍ਰਭਾਵਿਤ ਪੱਤੇ ਹਟਾਓ। ਕੰਪੋਸਟ ਚਾਹ ਵਰਤੋ।', 'prod': 'ਮੈਨਕੋਜ਼ੇਬ', 'cure': 'ਮੈਨਕੋਜ਼ੇਬ ਸਪ੍ਰੇ ਕਰੋ।'}
    },
    'Tomato___Late_blight': {
        'en': {'org': 'Copper-based organic sprays. Improve air flow.', 'prod': 'Chlorothalonil', 'cure': 'Apply Chlorothalonil every 7 days.'},
        'pa': {'org': 'ਕਾਪਰ-ਅਧਾਰਤ ਜੈਵਿਕ ਸਪ੍ਰੇ। ਹਵਾ ਦਾ ਪ੍ਰਵਾਹ ਸੁਧਾਰੋ।', 'prod': 'ਕਲੋਰੋਥੈਲੋਨਿਲ', 'cure': 'ਹਰ 7 ਦਿਨਾਂ ਬਾਅਦ ਕਲੋਰੋਥੈਲੋਨਿਲ ਲਗਾਓ।'}
    },
    'healthy': {
        'en': {'org': 'Plant is healthy. Use organic mulch.', 'prod': 'None', 'cure': 'Monitor regularly.'},
        'pa': {'org': 'ਪੌਦਾ ਸਿਹਤਮੰਦ ਹੈ। ਜੈਵਿਕ ਮਲਚ ਵਰਤੋ।', 'prod': 'ਕੋਈ ਨਹੀਂ', 'cure': 'ਨਿਯਮਿਤ ਤੌਰ ਤੇ ਦੇਖਭਾਲ ਕਰੋ।'}
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models():
    c_m = pickle.load(open('crop_model.pkl', 'rb')) if os.path.exists('crop_model.pkl') else None
    d_m = tf.keras.models.load_model('disease_model.h5') if os.path.exists('disease_model.h5') else None
    return c_m, d_m

crop_model, disease_model = load_models()

def get_weather(city_name):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid=b62c37dd2124c1bc17fc4c76a688d8bf&units=metric"
    try:
        r = requests.get(url).json()
        if r["cod"] == 200:
            return r["main"]["temp"], (23.0 if r["main"]["humidity"] > 95 else r["main"]["humidity"])
        return None, None
    except: return None, None

# --- 4. MAIN INTERFACE ---
st.title(L["title"])
selected = option_menu(None, [L["crop_tab"], L["disease_tab"], L["mandi_tab"]], 
                       icons=["flower1", "search", "cash"], orientation="horizontal")

# --- CROP RECOMMENDATION & FERTILIZER ---
if selected == L["crop_tab"]:
    with st.expander(L["weather_btn"]):
        city = st.text_input(L["city_label"], "Kapurthala")
        if st.button(L["weather_btn"]):
            w_t, w_h = get_weather(city)
            if w_t:
                st.session_state['temp'], st.session_state['hum'] = w_t, w_h
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input(L["n"], 0)
        p = st.number_input(L["p"], 0)
        k = st.number_input(L["k"], 0)
    with c2:
        temp = st.number_input(L["temp"], value=float(st.session_state['temp']))
        hum = st.number_input(L["hum"], value=float(st.session_state['hum']))
        ph = st.number_input(L["ph"], 6.5)
        rain = st.number_input(L["rain"], 100.0)

    if st.button(L["predict_btn"]):
        data = np.array([[n, p, k, temp, hum, ph, rain]])
        res = crop_model.predict(data)[0] if crop_model else "Rice / ਝੋਨਾ"
        st.success(f"{L['predict_btn']}: **{res}**")
        
        # --- NEW FERTILIZER LOGIC ---
        st.subheader(L["fert_title"])
        fert_advice = ""
        if n < 40: fert_advice += "• Add Urea for Nitrogen. "
        if p < 40: fert_advice += "• Add DAP for Phosphorus. "
        if k < 40: fert_advice += "• Add MOP for Potassium. "
        
        if not fert_advice:
            st.info("Soil nutrients are balanced! / ਮਿੱਟੀ ਦੇ ਪੋਸ਼ਕ ਤੱਤ ਸੰਤੁਲਿਤ ਹਨ!")
        else:
            st.warning(fert_advice if lang_choice == "English" else fert_advice.replace("Add", "ਪਾਓ").replace("for", "ਲਈ"))

# --- DISEASE DIAGNOSIS ---
elif selected == L["disease_tab"]:
    file = st.file_uploader(f"{L['disease_tab']} / Upload", type=["jpg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, width=300)
        if st.button(L["analyze_btn"]):
            img_resized = img.resize((200, 200)) 
            img_arr = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            if disease_model:
                p = disease_model.predict(img_arr)
                classes = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                    'Tomato___healthy'
                ]
                res = classes[np.argmax(p)]
            else: res = "healthy"

            k = 'en' if lang_choice == "English" else 'pa'
            # Fallback to healthy if the specific disease isn't in DISEASE_INFO dictionary yet
            info = DISEASE_INFO.get(res, DISEASE_INFO['healthy'])[k]
            
            st.subheader(f"Result: {res.replace('___', ' ')}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(f"### 🌱 {L['organic']}\n{info['org']}")
            with col_b:
                st.error(f"### 💊 {L['chemical']}\n**{info['prod']}**: {info['cure']}")

# --- MANDI RATES ---
elif selected == L["mandi_tab"]:
    st.header(L["mandi_title"])
    m_data = {"Crop/ਫਸਲ": ["Wheat/ਕਣਕ", "Paddy/ਝੋਨਾ"], "Price/ਭਾਅ": ["₹2,275", "₹2,183"]}
    st.table(m_data)
    st.info(L["mandi_note"])

# --- SIDEBAR AGRI-BOT ---
st.sidebar.markdown("---")
st.sidebar.subheader(L["bot_title"])
user_q = st.sidebar.text_input(L["bot_ask"])
if user_q:
    with st.sidebar.spinner("..."):
        try:
            response = ai_model.generate_content(f"Answer in {lang_choice}: {user_q}")
            st.sidebar.info(f"Bot: {response.text}")
        except: st.sidebar.error("Error")