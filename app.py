%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict if a delivery will be delayed.')

# Input fields for features
delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 50.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 10)
num_stops = st.slider('Number of Stops', 0, 10, 5)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 7)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 25.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 150, 75)

# Create a DataFrame for the input features
input_data = pd.DataFrame([[delivery_distance,
                            traffic_congestion,
                            weather_condition,
                            delivery_slot,
                            driver_experience,
                            num_stops,
                            vehicle_age,
                            road_condition_score,
                            package_weight,
                            fuel_efficiency,
                            warehouse_processing_time]],
                            columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                     'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                     'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                     'Warehouse_Processing_Time'])

if st.button('Predict Delivery Delay'):
    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error(f'Predicted: Delivery Delayed (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Predicted: No Delay (Probability: {prediction_proba[0][0]:.2f})')

    st.write('---')
    st.write('Feature values provided:')
    st.write(input_data)
