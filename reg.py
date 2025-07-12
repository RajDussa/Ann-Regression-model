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

# Example input fields (adjust based on original training features)
# You must use same order/columns as during model training
feature1 = st.number_input("Feature 1 (e.g., Age)", value=25.0)
feature2 = st.number_input("Feature 2 (e.g., Income)", value=50000.0)
feature3 = st.number_input("Feature 3 (e.g., Experience)", value=5.0)

# Add more features if required, matching the number and order used during training

# On button click
if st.button("Predict"):
    try:
        # Input vector (must match training feature order)
        input_data = np.array([[feature1, feature2, feature3]])  # shape (1, n_features)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0][0]

        # Display result
        st.subheader("Prediction:")
        st.write(f"Predicted Value: {prediction:.4f}")

    except Exception as e:
        st.error(f"Error occurred: {e}")
