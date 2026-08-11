import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/catboost.pkl")

# page configuration

st.set_page_config(
    page_title = "Flight Price Prediction",
    page_icon = "✈️",
    layout = "wide"
)

st.title("Flight Price Prediction System")
st.write(
    "Enter the flight details below to estimate the ticket price."
)

# user inputs

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline",
        [
            "indigo",
            "air india",
            'vistara',
            "airasia india",
            "akasa air"
        ]
    )

    source_city = st.selectbox(
        "Source City",
        [
            "delhi",
            "mumbai",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad"
        ]
    )

    destination_city = st.selectbox(
        "Destination City",
        [
            "delhi",
            "mumbai",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad"
        ]
    )

    total_stops = st.number_input(
        "Total Stops",
        min_value=0,
        max_value=3,
        value=0
    )

    travel_class = st.selectbox(
        "Travel Class",
        ["economy", "business"]
    )

    booking_window_days = st.slider(
        "Booking Window (Days)",
        min_value=1,
        max_value=90,
        value=30
    )

    season = st.selectbox(
        "Season",
        ["summer", "monsoon", "winter", "spring"]
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]
    )

with col2:

    is_holiday = st.selectbox(
        "Holiday",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    aircraft_type = st.selectbox(
        "Aircraft Type",
        [
            "A320",
            "A321",
            "A350",
            "B737",
            "B787",
            "ATR72"
        ]
    )

    distance_km = st.number_input(
        "Distance (km)",
        min_value=100,
        max_value=6000,
        value=800
    )

    rating = st.slider(
        "Airline Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    duration_minutes = st.number_input(
        "Duration (minutes)",
        min_value=30,
        max_value=1000,
        value=120
    )

    journey_month = st.slider(
        "Journey Month",
        min_value=1,
        max_value=12,
        value=8
    )

    journey_day = st.slider(
        "Journey Day",
        min_value=1,
        max_value=31,
        value=15
    )

    departure_hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=10
    )

    arrival_hour = st.slider(
        "Arrival Hour",
        min_value=0,
        max_value=23,
        value=12
    )

# Prediction

if st.button("Predict Flight Price", type="primary"):

    # The column names must match the training features.

    # Validate the route
    if source_city == destination_city:
        st.warning(
            "Source and destination cities cannot be the same. Please select different cities."
        )

    else:
        # Create a single-row DataFrame
        input_data = pd.DataFrame({
            "airline": [airline],
            "source_city": [source_city],
            "destination_city": [destination_city],
            "total_stops": [total_stops],
            "travel_class": [travel_class],
            "booking_window_days": [booking_window_days],
            "season": [season],
            "day_of_week": [day_of_week],
            "is_holiday": [is_holiday],
            "aircraft_type": [aircraft_type],
            "distance_km": [distance_km],
            "rating": [rating],
            "duration_minutes": [duration_minutes],
            "journey_month": [journey_month],
            "journey_day": [journey_day],
            "departure_hour": [departure_hour],
            "arrival_hour": [arrival_hour],
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Estimated Flight Price: ₹{prediction:,.0f}"
        )

# Sidebar

st.sidebar.title("Model Information")
st.sidebar.write("**Deployed Model:** CatBoost Regressor")
st.sidebar.write("**R² Score:** 0.68")
st.sidebar.write("**MAE:** ₹1,293")
st.sidebar.write("**RMSE:** ₹3,324")