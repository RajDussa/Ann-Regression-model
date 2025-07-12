import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("regression_model.h5", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit App UI
st.title("Regression Prediction App")

st.header("Enter Input Features")

# Updated: 5 input fields (as scaler was trained on 5 features)
feature1 = st.number_input("Feature 1", value=25.0)
feature2 = st.number_input("Feature 2", value=50000.0)
feature3 = st.number_input("Feature 3", value=5.0)
feature4 = st.number_input("Feature 4", value=1.0)
feature5 = st.number_input("Feature 5", value=100.0)

# Predict on button click
if st.button("Predict"):
    try:
        # Combine features into array matching the training input format
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])

        # Scale input using the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"Predicted Value: {prediction:.4f}")

    except Exception as e:
        st.error(f"Error occurred: {e}")
