import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
MODEL_PATH = "svm_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please ensure the model is present in the correct path.")
    st.stop()

st.set_page_config(page_title="Stress Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Stress Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter your physiological details to predict stress levels</h4>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About This App")
st.sidebar.info("This app predicts stress levels based on physiological parameters using a trained ML model.")

# Liberalized Ranges for Validation
ideal_ranges = {
    "snoring_range": (0, 120),  # Extended range
    "respiration_rate": (10, 30),
    "body_temp": (95, 101),
    "limb_movement": (0, 20),
    "blood_oxygen": (90, 100),
    "eye_movement": (0, 100),
    "sleep_hours": (1, 24),
    "heart_rate": (50, 110)
}

# Two-column layout for input fields
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    snoring_range = st.text_input("Snoring Range", "0.0")
    respiration_rate = st.text_input("Respiration Rate", "0.0")
    body_temp = st.text_input("Body Temperature (F)", "0.0")
    limb_movement = st.text_input("Limb Movement Rate", "0.0")

# Column 2 inputs
with col2:
    blood_oxygen = st.text_input("Blood Oxygen Levels (%)", "0.0")
    eye_movement = st.text_input("Eye Movement Rate", "0.0")
    sleep_hours = st.text_input("Sleep Hours", "0.0")
    heart_rate = st.text_input("Heart Rate (BPM)", "0.0")

# Button to Predict Stress Level
if st.button("Predict Stress Level"):
    try:
        # Convert inputs to float and validate ranges
        inputs = {
            "snoring_range": float(snoring_range),
            "respiration_rate": float(respiration_rate),
            "body_temp": float(body_temp),
            "limb_movement": float(limb_movement),
            "blood_oxygen": float(blood_oxygen),
            "eye_movement": float(eye_movement),
            "sleep_hours": float(sleep_hours),
            "heart_rate": float(heart_rate)
        }
        
        warnings = []
        for key, (low, high) in ideal_ranges.items():
            if not (low <= inputs[key] <= high):
                warnings.append(f"{key.replace('_', ' ').title()} is out of the ideal range ({low}-{high})! Please enter a valid value.")
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
            st.error("Cannot predict stress level due to invalid inputs. Please provide values within the valid ranges.")
        else:
            # Create feature array
            features = pd.DataFrame([[
                inputs["snoring_range"], inputs["respiration_rate"], inputs["body_temp"], inputs["limb_movement"],
                inputs["blood_oxygen"], inputs["eye_movement"], inputs["sleep_hours"], inputs["heart_rate"]
            ]], columns=["sr", "rr", "t", "lm", "bo", "rem", "sh", "hr"])
            
            # Model prediction
            prediction = model.predict(features)[0]
            
            # Display the prediction result
            st.markdown(f"""
            <div style="background-color:#4CAF50;padding:10px;border-radius:10px;">
                <h3 style="text-align:center;color:white;">Predicted Stress Level: {prediction}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    except ValueError as e:
        st.error(f"Error: {str(e)}. Please ensure all fields contain valid numeric values.")
