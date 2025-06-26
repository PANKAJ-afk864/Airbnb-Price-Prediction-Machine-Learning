import streamlit as st
import joblib
import numpy as np

model = joblib.load('airbnb_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🏡 Airbnb Price Predictor (NYC)")

latitude = st.number_input("Latitude", value=40.7)
longitude = st.number_input("Longitude", value=-73.9)
min_nights = st.slider("Minimum Nights", 1, 365, 3)
num_reviews = st.slider("Number of Reviews", 0, 500, 10)
reviews_per_month = st.number_input("Reviews per Month", value=1.0)
host_listings = st.slider("Host Listings Count", 1, 50, 1)
availability = st.slider("Availability (days/year)", 0, 365, 180)

room_type = st.selectbox("Room Type", ["Private room", "Shared room", "Entire home/apt"])
room_dict = {
    "Private room": [1, 0],
    "Shared room": [0, 1],
    "Entire home/apt": [0, 0]
}

borough = st.selectbox("Neighbourhood Group", ["Brooklyn", "Manhattan", "Queens", "Staten Island", "Bronx"])
borough_dict = {
    "Brooklyn": [1, 0, 0, 0],
    "Manhattan": [0, 1, 0, 0],
    "Queens": [0, 0, 1, 0],
    "Staten Island": [0, 0, 0, 1],
    "Bronx": [0, 0, 0, 0]
}

input_data = np.array([
    latitude, longitude, min_nights, num_reviews,
    reviews_per_month, host_listings, availability,
    *borough_dict[borough], *room_dict[room_type]
]).reshape(1, -1)

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Price: ${round(prediction)} per night")
