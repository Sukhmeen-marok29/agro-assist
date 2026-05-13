import os 
import sys
import numpy as np
import streamlit as st
import pickle
import tensorflow as tf
import requests
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import datetime

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="Agri-Smart Pro", layout="wide")

if 'temp' not in st.session_state: st.session_state['temp'] = 0.0
if 'hum' not in st.session_state: st.session_state['hum'] = 0.0
if 'rain' not in st.session_state: st.session_state['rain'] = 0.0

# Language Dictionary
LANG_DICT = {
    "English": {
        "title": "AgriPulse AI: Smart Farming Assistant",
        "crop_tab": "Crop Recommendation",
        "disease_tab": "Disease Diagnosis",
        "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "ph": "Soil pH", "rain": "Rainfall (mm)",
        "predict_btn": "Predict Best Crop",
        "weather_btn": "Fetch Current Weather",
        "city_label": "Enter City Name",
        "analyze_btn": "Analyze Image",
        "organic": "Organic Cure",
        "chemical": "Chemical Treatment",
        "fert_title": "💡 Fertilizer Suggestion",
        "sowing_title": "📅 Sowing Schedule",
        "current_month": "Current Month",
        "accuracy_label": "Model Confidence"
    },
    "ਪੰਜਾਬੀ": {
        "title": "🌱 ਐਗਰੀ-ਸਮਾਰਟ ਪ੍ਰੋ",
        "crop_tab": "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        "disease_tab": "ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
        "n": "ਨਾਈਟ੍ਰੋਜਨ (N)", "p": "ਫਾਸਫੋਰਸ (P)", "k": "ਪੋਟਾਸ਼ੀਅਮ (K)",
        "temp": "ਤਾਪਮਾਨ (°C)", "hum": "ਨਮੀ (%)", "ph": "ਮਿੱਟੀ ਦਾ pH", "rain": "ਵਰਖਾ (mm)",
        "predict_btn": "ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਦੇਖੋ",
        "weather_btn": "ਤਾਜ਼ਾ ਮੌਸਮ ਦੇਖੋ",
        "city_label": "ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਲਿਖੋ",
        "analyze_btn": "ਜਾਂਚ ਕਰੋ",
        "organic": "ਜੈਵਿਕ ਇਲਾਜ",
        "chemical": "ਰਸਾਇਣਕ ਇਲਾਜ",
        "fert_title": "💡 ਖਾਦ ਦੀ ਸਲਾਹ",
        "sowing_title": "📅 ਬਿਜਾਈ ਦਾ ਸਮਾਂ",
        "current_month": "ਮੌਜੂਦਾ ਮਹੀਨਾ",
        "accuracy_label": "ਮਾਡਲ ਦਾ ਭਰੋਸਾ"
    }
}

st.sidebar.title("Settings / ਸੈਟਿੰਗਾਂ")
lang_choice = st.sidebar.radio("Select Language", ["English", "ਪੰਜਾਬੀ"])
L = LANG_DICT[lang_choice]

# --- 2. KNOWLEDGE BASE & SOWING DATA ---
SOWING_WINDOWS = {
    "Wheat": {"start": 10, "end": 11, "name_pa": "ਕਣਕ"},
    "Rice": {"start": 6, "end": 7, "name_pa": "ਝੋਨਾ"},
    "Cotton": {"start": 4, "end": 5, "name_pa": "ਨਰਮਾ"},
    "Maize": {"start": 6, "end": 6, "name_pa": "ਮੱਕੀ"},
    "Mustard": {"start": 10, "end": 11, "name_pa": "ਸਰ੍ਹੋਂ"},
    "Sugarcane": {"start": 2, "end": 3, "name_pa": "ਗੰਨਾ"},
    "Potato": {"start": 9, "end": 10, "name_pa": "ਆਲੂ"},
    "Muskmelon": {"start": 2, "end": 3, "name_pa": "ਖਰਬੂਜਾ"},
    "Watermelon": {"start": 2, "end": 3, "name_pa": "ਤਰਬੂਜ"},
    "Mungbean": {"start": 3, "end": 4, "name_pa": "ਮੂੰਗੀ"},
    "Pomegranate": {"start": 7, "end": 8, "name_pa": "ਅਨਾਰ"}
}
# EXPANDED DISEASE INFO from 752c8e6d-a41c-48d4-9cb5-a1f11043922c
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'en': {'org': 'Apply Neem oil or baking soda spray.', 'prod': 'Captan 50 WP', 'cure': 'Apply every 7-10 days.'},
        'pa': {'org': 'ਨੀਮ ਦਾ ਤੇਲ ਜਾਂ ਬੇਕਿੰਗ ਸੋਡਾ ਸਪ੍ਰੇ ਕਰੋ।', 'prod': 'ਕੈਪਟਨ 50 WP', 'cure': 'ਹਰ 7-10 ਦਿਨਾਂ ਬਾਅਦ ਲਾਗੂ ਕਰੋ।'}
    },
    'Apple___Black_rot': {
        'en': {'org': 'Remove mummified fruit and prune dead wood.', 'prod': 'Copper-based fungicides', 'cure': 'Prune in winter.'},
        'pa': {'org': 'ਸੁੱਕੇ ਫਲਾਂ ਨੂੰ ਹਟਾਓ ਅਤੇ ਮਰੀ ਹੋਈ ਲੱਕੜ ਨੂੰ ਕੱਟੋ।', 'prod': 'ਕਾਪਰ-ਅਧਾਰਤ ਉੱਲੀਨਾਸ਼ਕ', 'cure': 'ਸਰਦੀਆਂ ਵਿੱਚ ਕਾਂਟ-ਛਾਂਟ ਕਰੋ।'}
    },
    'Apple___Cedar_apple_rust': {
        'en': {'org': 'Remove nearby Juniper trees.', 'prod': 'Myclobutanil', 'cure': 'Spray at blossom.'},
        'pa': {'org': 'ਨੇੜਲੇ ਜੂਨੀਪਰ ਦੇ ਰੁੱਖਾਂ ਨੂੰ ਹਟਾਓ।', 'prod': 'ਮਾਈਕਲੋਬੂਟਾਨਿਲ', 'cure': 'ਫੁੱਲ ਆਉਣ ਤੇ ਸਪ੍ਰੇ ਕਰੋ।'}
    },
    'Potato___Early_blight': {
        'en': {'org': 'Remove infected leaves. Use compost tea.', 'prod': 'Mancozeb', 'cure': 'Avoid overhead irrigation.'},
        'pa': {'org': 'ਪ੍ਰਭਾਵਿਤ ਪੱਤੇ ਹਟਾਓ। ਕੰਪੋਸਟ ਚਾਹ ਵਰਤੋ।', 'prod': 'ਮੈਨਕੋਜ਼ੇਬ', 'cure': 'ਉੱਪਰੋਂ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ।'}
    },
    'Potato___Late_blight': {
        'en': {'org': 'Ensure dry foliage and improve air circulation.', 'prod': 'Chlorothalonil', 'cure': 'Destroy cull piles.'},
        'pa': {'org': 'ਪੱਤਿਆਂ ਨੂੰ ਸੁੱਕਾ ਰੱਖੋ ਅਤੇ ਹਵਾ ਦਾ ਪ੍ਰਵਾਹ ਸੁਧਾਰੋ।', 'prod': 'ਕਲੋਰੋਥੈਲੋਨਿਲ', 'cure': 'ਢੇਰਾਂ ਨੂੰ ਨਸ਼ਟ ਕਰੋ।'}
    },
    'Tomato___Late_blight': {
        'en': {'org': 'Copper-based organic sprays. Improve air flow.', 'prod': 'Chlorothalonil', 'cure': 'Apply immediately.'},
        'pa': {'org': 'ਕਾਪਰ-ਅਧਾਰਤ ਜੈਵਿਕ ਸਪ੍ਰੇ। ਹਵਾ ਦਾ ਪ੍ਰਵਾਹ ਸੁਧਾਰੋ।', 'prod': 'ਕਲੋਰੋਥੈਲੋਨਿਲ', 'cure': 'ਤੁਰੰਤ ਲਾਗੂ ਕਰੋ।'}
    },
    'Tomato___Target_Spot': {
        'en': {'org': 'Apply mulch to prevent soil splashing.', 'prod': 'Azoxystrobin', 'cure': 'Improve spacing.'},
        'pa': {'org': 'ਮਿੱਟੀ ਦੇ ਛਿੱਟੇ ਰੋਕਣ ਲਈ ਮਲਚ ਦੀ ਵਰਤੋਂ ਕਰੋ।', 'prod': 'ਅਜ਼ੋਕਸੀਸਟ੍ਰੋਬਿਨ', 'cure': 'ਫਾਸਲਾ ਸੁਧਾਰੋ।'}
    },
    'healthy': {
        'en': {'org': 'Plant is healthy. Use organic mulch.', 'prod': 'None', 'cure': 'Maintain regular care.'},
        'pa': {'org': 'ਪੌਦਾ ਸਿਹਤਮੰਦ ਹੈ। ਜੈਵਿਕ ਮਲਚ ਵਰਤੋ।', 'prod': 'ਕੋਈ ਨਹੀਂ', 'cure': 'ਨਿਯਮਤ ਦੇਖਭਾਲ ਰੱਖੋ।'}
    }
}

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_models():
    c_m = pickle.load(open('crop_model.pkl', 'rb')) if os.path.exists('crop_model.pkl') else None
    d_m = tf.keras.models.load_model('disease_model.h5') if os.path.exists('disease_model.h5') else None
    return c_m, d_m

crop_model, disease_model = load_models()

def get_weather(city_name):
    api_key = "b62c37dd2124c1bc17fc4c76a688d8bf"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try: 
        r = requests.get(url).json()
        if r.get("cod") == 200:
            return r["main"]["temp"], r["main"]["humidity"], r.get("rain", {}).get("1h", 0.0)
    except Exception as e:
        st.error(f"Weather error: {e}")
    return None, None, None       

def get_sowing_advice(crop_name, lang):
    clean_name = crop_name.split(" / ")[0].strip()
    current_month = datetime.datetime.now().month
    if clean_name not in SOWING_WINDOWS:
        return{"status":"No Data", "msg": "Sowing info not available.", "color": "gray"} 
    
    win = SOWING_WINDOWS[clean_name]
    if win["start"] <= current_month <= win["end"]:
        status, color = ("✅ IDEAL TIME", "green") if lang == "English" else ("✅ ਸਹੀ ਸਮਾਂ", "green")
        msg = f"Perfect time to sow {clean_name}." if lang == "English" else f"ਹੁਣ {win['name_pa']} ਬੀਜਣ ਦਾ ਸਹੀ ਸਮਾਂ ਹੈ।"
    elif current_month < win["start"]:
        status, color = ("⏳ TOO EARLY", "blue") if lang == "English" else ("⏳ ਬਹੁਤ ਜਲਦੀ", "blue")
        msg = f"Wait until month {win['start']}." if lang == "English" else f"ਮਹੀਨੇ {win['start']} ਤੱਕ ਉਡੀਕ ਕਰੋ।"
    else:
        status, color = ("⚠️ LATE SOWING", "orange") if lang == "English" else ("⚠️ ਪਛੇਤੀ ਬਿਜਾਈ", "orange")
        msg = f"Yield might be affected." if lang == "English" else f"ਝਾੜ ਘਟ ਸਕਦਾ ਹੈ।"
    return {"status": status, "msg": msg, "color": color}

# --- 4. MAIN INTERFACE ---
st.title(L["title"])
st.markdown("---")

selected = option_menu(None, [L["crop_tab"], L["disease_tab"]], 
                       icons=["flower1", "search"], orientation="horizontal")

if selected == L["crop_tab"]:
    with st.expander(L["weather_btn"]):
        city = st.text_input(L["city_label"], "Kapurthala")
        if st.button(L["weather_btn"]):
            w_t, w_h, w_r = get_weather(city)
            if w_t is not None:
                st.session_state['temp'], st.session_state['hum'], st.session_state['rain'] = w_t, w_h, w_r
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input(L["n"], min_value=0, value=0)
        p = st.number_input(L["p"], min_value=0, value=0)
        k = st.number_input(L["k"], min_value=0, value=0)
    with c2:
        temp = st.number_input(L["temp"], value=float(st.session_state['temp']))
        hum = st.number_input(L["hum"], value=float(st.session_state['hum']))
        ph = st.number_input(L["ph"], value=6.5)
        rain = st.number_input(L["rain"], value=float(st.session_state['rain']))

    if st.button(L["predict_btn"]):
        if crop_model:
            data = np.array([[n, p, k, temp, hum, ph, rain]])
            res = crop_model.predict(data)[0]
        else: res = "Rice"
        
        st.success(f"### {L['predict_btn']}: **{res}**")
        
        advice = get_sowing_advice(res, lang_choice)
        if advice:
            st.markdown("---")
            st.subheader(L["sowing_title"])
            st.markdown(f"<h3 style='color:{advice['color']};'>{advice['status']}</h3>", unsafe_allow_html=True)
            st.info(advice['msg'])

elif selected == L["disease_tab"]:
    file = st.file_uploader(L["disease_tab"], type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=300)
        if st.button(L["analyze_btn"]):
            if disease_model:
                img_resized = img.resize((200, 200))
                img_arr = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                prediction = disease_model.predict(img_arr)
                
                # --- FULL LIST OF 38 CLASSES ---
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
                
                # Accuracy Calculation from 752c8e6d-a41c-48d4-9cb5-a1f11043922c
                result_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                res = classes[result_index]
                
                # Display Accuracy
                st.metric(label=L["accuracy_label"], value=f"{confidence:.2f}%")
                st.progress(int(confidence))
            else: 
                res = "healthy"
                confidence = 100.0

            k = 'en' if lang_choice == "English" else 'pa'
            # Safe lookup with fallback
            info = DISEASE_INFO.get(res, DISEASE_INFO['healthy'])[k]
            
            st.subheader(f"Result: {res.replace('___', ' ')}")
            ca, cb = st.columns(2)
            ca.success(f"### 🌱 {L['organic']}\n{info['org']}")
            cb.error(f"### 💊 {L['chemical']}\n**{info['prod']}**: {info['cure']}")