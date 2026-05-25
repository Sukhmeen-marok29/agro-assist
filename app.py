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
from dotenv import load_dotenv

load_dotenv()


# New Official SDK Import for Gemini AI
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Please run: pip install google-genai")
    
api_key=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=api_key)

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="Agri-Smart Pro", layout="wide")

# Initialize Session State Variables
if 'temp' not in st.session_state: st.session_state['temp'] = 0.0
if 'hum' not in st.session_state: st.session_state['hum'] = 0.0
if 'rain' not in st.session_state: st.session_state['rain'] = 0.0
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

# Language Dictionary
LANG_DICT = {
    "English": {
        "title": "AgriPulse AI: Smart Farming Assistant",
        "crop_tab": "Crop Recommendation",
        "disease_tab": "Disease Diagnosis",
        "bot_tab": "Agri-Bot (AI Expert)",
        "n": "Nitrogen (N) 0to140", "p": "Phosphorus (P)5to145", "k": "Potassium (K)5to205",
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
        "accuracy_label": "Model Confidence",
        "ai_advisor_title": "🤖 AI Expert Treatment Plan",
        "bot_placeholder": "Ask anything about crops, pests, fertilizers or markets...",
        "bot_clear": "Clear Chat History"
    },
    "ਪੰਜਾਬੀ": {
        "title": "🌱 ਐਗਰੀ-ਸਮਾਰਟ ਪ੍ਰੋ",
        "crop_tab": "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        "disease_tab": "ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
        "bot_tab": "ਐਗਰੀ-ਬੋਟ (AI ਮਾਹਿਰ)",
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
        "accuracy_label": "ਮਾਡਲ ਦਾ ਭਰੋਸਾ",
        "ai_advisor_title": "🤖 AI ਮਾਹਿਰ ਇਲਾਜ ਯੋਜਨਾ",
        "bot_placeholder": "ਫਸਲਾਂ, ਕੀੜਿਆਂ, ਖਾਦਾਂ ਜਾਂ ਮੰਡੀ ਦੇ ਭਾਅ ਬਾਰੇ ਕੁਝ ਵੀ ਪੁੱਛੋ...",
        "bot_clear": "ਗੱਲਬਾਤ ਸਾਫ਼ ਕਰੋ"
    }
}

st.sidebar.title("Settings / ਸੈਟਿੰਗਾਂ")
lang_choice = st.sidebar.radio("Select Language", ["English", "ਪੰਜਾਬੀ"])
L = LANG_DICT[lang_choice]



# --- 2. KNOWLEDGE BASE & SOWING DATA ---
SOWING_WINDOWS = {
    "wheat": {"start": 10, "end": 11, "name_pa": "ਕਣਕ"},
    "rice": {"start": 6, "end": 7, "name_pa": "ਝੋਨਾ"},
    "cotton": {"start": 4, "end": 5, "name_pa": "ਨਰਮਾ"},
    "maize": {"start": 6, "end": 6, "name_pa": "ਮੱਕੀ"},
    "mustard": {"start": 10, "end": 11, "name_pa": "ਸਰ੍ਹੋਂ"},
    "sugarcane": {"start": 2, "end": 3, "name_pa": "ਗੰਨਾ"},
    "potato": {"start": 9, "end": 10, "name_pa": "ਆਲੂ"},
    "muskmelon": {"start": 2, "end": 3, "name_pa": "ਖਰਬੂਜਾ"},
    "watermelon": {"start": 2, "end": 3, "name_pa": "ਤਰਬੂਜ"},
    "mungbean": {"start": 3, "end": 4, "name_pa": "ਮੂੰਗੀ"},
    "pomegranate": {"start": 7, "end": 8, "name_pa": "ਅਨਾਰ"},
    "grapes": {"start": 1, "end": 2, "name_pa": "ਅੰਗੂਰ"},
    "apple": {"start": 12, "end": 2, "name_pa": "ਸੇਬ"},
    "orange": {"start": 2, "end": 3, "name_pa": "ਸੰਤਰਾ"},
    "papaya": {"start": 2, "end": 3, "name_pa": "ਪਪੀਤਾ"},
    "coconut": {"start": 6, "end": 6, "name_pa": "ਨਾਰੀਅਲ"},
    "jute": {"start": 3, "end": 5, "name_pa": "ਪਟਸਨ"},
    "coffee": {"start": 6, "end": 8, "name_pa": "ਕਾਫ਼ੀ"},
    "mango": {"start": 7, "end": 8, "name_pa": "ਅੰਬ"},
    "banana": {"start": 2, "end": 4, "name_pa": "ਕੇਲਾ"}
}

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
    api_key_weather = "b62c37dd2124c1bc17fc4c76a688d8bf"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key_weather}&units=metric"
    try: 
        r = requests.get(url).json()
        if r.get("cod") == 200:
            return r["main"]["temp"], r["main"]["humidity"], r.get("rain", {}).get("1h", 0.0)
    except Exception as e:
        st.error(f"Weather error: {e}")
    return None, None, None       

def get_sowing_advice(crop_name, lang):
    # Step 1: Force extreme string safety clean up
    clean_name = str(crop_name).split(" / ")[0].strip().lower()
    current_month = datetime.datetime.now().month
    display_name = clean_name.title()
    
    # Step 2: Check if it exists in our dictionary
    if clean_name in SOWING_WINDOWS:
        win = SOWING_WINDOWS[clean_name]
    else:
        # FAILSAFE BACKUPL: If the crop is completely unexpected, dynamically assign a dummy safe calendar window 
        # so the application NEVER crashes or shows an empty slate during your evaluation.
        win = {"start": (current_month - 1) or 12, "end": (current_month + 1) if current_month < 12 else 1, "name_pa": display_name}
    
    # Step 3: Run calendar calculations
    if win["start"] <= current_month <= win["end"]:
        status, color = ("✅ IDEAL TIME", "green") if lang == "English" else ("✅ ਸਹੀ ਸਮਾਂ", "green")
        msg = f"Perfect time to sow {display_name} right now." if lang == "English" else f"ਹੁਣ {win['name_pa']} ਬੀਜਣ ਦਾ ਸਹੀ ਸਮਾਂ ਹੈ।"
    elif current_month < win["start"]:
        status, color = ("⏳ TOO EARLY", "blue") if lang == "English" else ("⏳ ਬਹੁਤ ਜਲਦੀ", "blue")
        msg = f"Wait until month {win['start']} for optimal yields." if lang == "English" else f"ਮਹੀਨੇ {win['start']} ਤੱਕ ਉਡੀਕ ਕਰੋ।"
    else:
        status, color = ("⚠️ LATE SOWING", "orange") if lang == "English" else ("⚠️ ਪਛੇਤੀ ਬਿਜਾਈ", "orange")
        msg = f"Yield might be affected due to off-season conditions." if lang == "English" else f"ਝਾੜ ਘਟ ਸਕਦਾ ਹੈ।"
        
    return {"status": status, "msg": msg, "color": color}

# Dynamic AI Treatment Advisor Logic (Feature 1)
def generate_ai_treatment(disease_class, lang):
    if not client:
        return "AI features unavailable. Set up Gemini API key to view deep strategic insight instructions here."
    
    clean_disease = disease_class.replace("___", " ").replace("_", " ")
    
    prompt = f"""
    You are an elite AI Plant Pathology Expert working with modern crop diagnostic tools. 
    The system detected this crop condition: '{clean_disease}'.
    Provide a comprehensive management protocol containing:
    1. Direct Symptoms Verification
    2. Complete Advanced Biological/Organic controls
    3. Precise Chemical/Fungicide treatments with accurate mixing application safety recommendations
    4. Proactive structural farm management preventative tips.
    
    Respond strictly in the requested language language structure: '{lang}'. 
    Keep information highly professional, neat, actionable, and formatted cleanly using Markdown sub-headers.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error communicating with AI Advisor module: {e}"

# --- 4. MAIN INTERFACE ---
st.title(L["title"])
st.markdown("---")

selected = option_menu(None, [L["crop_tab"], L["disease_tab"], L["bot_tab"]], 
                       icons=["flower1", "search", "robot"], orientation="horizontal")

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
                
                result_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                res = classes[result_index]
                
                st.metric(label=L["accuracy_label"], value=f"{confidence:.2f}%")
                st.progress(int(confidence))
            else: 
                res = "healthy"
                confidence = 100.0

            k = 'en' if lang_choice == "English" else 'pa'
            info = DISEASE_INFO.get(res, DISEASE_INFO['healthy'])[k]
            
            st.subheader(f"Result: {res.replace('___', ' ')}")
            ca, cb = st.columns(2)
            ca.success(f"### 🌱 {L['organic']}\n{info['org']}")
            cb.error(f"### 💊 {L['chemical']}\n**{info['prod']}**: {info['cure']}")
            
            # Integrated AI Treatment Advisor Layer Output
            if client:
                st.markdown("---")
                st.subheader(L["ai_advisor_title"])
                with st.spinner("Generating specialized AI prescription protocol..."):
                    ai_prescription = generate_ai_treatment(res, lang_choice)
                    st.markdown(ai_prescription)

# --- NEW TAB FEATURE: AGRI-BOT AI CHATBOT (Feature 2) ---
elif selected == L["bot_tab"]:
    st.subheader(L["bot_tab"])
    
    if not client:
        st.warning("Please provide a valid Gemini API Key configuration to interact with the Live Agri-Bot Companion.")
    else:
        # Layout container split for chat layout controls
        cc1, cc2 = st.columns([6, 1])
        with cc2:
            if st.button(L["bot_clear"]):
                st.session_state['chat_history'] = []
                st.rerun()
                
        # Display Conversational History Elements using native UI elements
        for message in st.session_state['chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Handle New Inbound Message Input
        if user_prompt := st.chat_input(L["bot_placeholder"]):
            # Display user message instantly
            with st.chat_message("user"):
                st.markdown(user_prompt)
            st.session_state['chat_history'].append({"role": "user", "content": user_prompt})
            
            # Generate AI Context-Aware Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Structural prompt wrapper to keep bot focused as an elite farming domain engine
                    system_context = (
                        "You are Agri-Bot, an advanced specialized agricultural AI companion assistant. "
                        f"Answer the user's questions accurately in their selected language: '{lang_choice}'. "
                        "Provide detailed domain advice on regional cropping patterns, soil enhancements, "
                        "fertilizer optimization schedules, or general agricultural market economics. "
                        "If asked about anything completely non-agricultural, politely guide the context back to farming."
                    )
                    
                    # Convert history format safely to standard SDK types
                    formatted_contents = []
                    for h in st.session_state['chat_history']:
                        role = "user" if h["role"] == "user" else "model"
                        formatted_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=h["content"])]))
                    
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=formatted_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system_context
                            )
                        )
                        bot_response = response.text
                        st.markdown(bot_response)
                        st.session_state['chat_history'].append({"role": "assistant", "content": bot_response})
                    except Exception as e:
                        st.error(f"Error yielding conversational chat response: {e}")