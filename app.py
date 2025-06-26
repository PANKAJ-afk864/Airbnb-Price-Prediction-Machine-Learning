import streamlit as st
import numpy as np
import joblib

# Load model and features
model = joblib.load('airbnb_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🏡 Airbnb NYC Price Predictor")

# ---- User Inputs ----
latitude = st.number_input("Latitude", value=40.7)
longitude = st.number_input("Longitude", value=-73.9)
min_nights = st.slider("Minimum Nights", 1, 365, 3)
num_reviews = st.slider("Number of Reviews", 0, 500, 10)
reviews_per_month = st.number_input("Reviews per Month", value=1.0)
host_listings = st.slider("Host Listings Count", 1, 50, 1)
availability = st.slider("Availability (days/year)", 0, 365, 180)

room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
borough = st.selectbox("Neighbourhood Group", ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])

# ---- Feature Alignment ----
input_dict = {
    'latitude': latitude,
    'longitude': longitude,
    'minimum_nights': min_nights,
    'number_of_reviews': num_reviews,
    'reviews_per_month': reviews_per_month,
    'calculated_host_listings_count': host_listings,
    'availability_365': availability,
    'neighbourhood_group_Brooklyn': 0,
    'neighbourhood_group_Manhattan': 0,
    'neighbourhood_group_Queens': 0,
    'neighbourhood_group_Staten Island': 0,
    'room_type_Private room': 0,
    'room_type_Shared room': 0,
}

# Set appropriate one-hot encoding
if f'neighbourhood_group_{borough}' in input_dict:
    input_dict[f'neighbourhood_group_{borough}'] = 1

if room_type != 'Entire home/apt':  # Entire home/apt is the dropped base
    input_dict[f'room_type_{room_type}'] = 1

# Ensure correct feature order
input_list = [input_dict.get(col, 0) for col in features]
input_array = np.array(input_list).reshape(1, -1)

# ---- Predict ----
if st.button("Predict Price"):
    prediction = model.predict(input_array)
    st.success(f"Predicted Price: ${prediction[0]:.2f}")    
