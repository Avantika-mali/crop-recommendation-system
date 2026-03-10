import streamlit as st
import pandas as pd
import pickle

# 1. Use caching to load the model and scaler only once
@st.cache_resource
def load_assets():
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")
st.title("🌱 Crop Recommendation System")
st.write("Provide the following soil and environmental parameters to get a recommendation.")

# 2. Use columns for a cleaner UI layout
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, value=50)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, value=25.0)

with col2:
    P = st.number_input("Phosphorus (P)", 0, 200, value=50)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, value=50.0)

with col3:
    K = st.number_input("Potassium (K)", 0, 200, value=50)
    ph = st.number_input("pH Value", 0.0, 14.0, value=6.5)

rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, value=100.0)

if st.button("Predict Best Crop", use_container_width=True):
    # Ensure this list order matches your training CSV exactly!
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                              columns=feature_names)

    # Transform and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    # Visual feedback
    st.markdown("---")
    st.success(f"### The best crop for these conditions is: **{prediction[0].upper()}**")