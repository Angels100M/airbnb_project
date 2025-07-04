import streamlit as st
import pandas as pd
import joblib

# טען את המודל המאומן
model = joblib.load("models/price_predictor_xgb.pkl")

# טען את שמות כל הפיצ'רים מהקובץ שיצרת
feature_columns = pd.read_csv('data/X_train.csv').columns.tolist()

# דוגמא לערכי ברירת מחדל
default_values = {
    'latitude': 37.77,
    'longitude': -122.42,
    'accommodates': 2,
    'bathrooms': 1.0,        
    'bedrooms': 1.0,         
    'beds': 1.0,             
    'property_type': 'Entire home',
    'room_type': 'Entire home/apt',
    'neighbourhood_cleansed': 'Downtown/Civic Center'
}

# הפקת רשימות לכל התפריטים
property_types = sorted([c.replace('property_type_', '') for c in feature_columns if c.startswith('property_type_')])
room_types = sorted([c.replace('room_type_', '') for c in feature_columns if c.startswith('room_type_')])
neighbourhoods = sorted([c.replace('neighbourhood_cleansed_', '') for c in feature_columns if c.startswith('neighbourhood_cleansed_')])

st.title("Airbnb Price Prediction App")

with st.form("input_form"):
    st.header("הזן פרטי דירה")
    latitude = st.number_input("Latitude", value=default_values['latitude'])
    longitude = st.number_input("Longitude", value=default_values['longitude'])
    accommodates = st.number_input("Accommodates (מספר אורחים)", min_value=1, max_value=20, value=default_values['accommodates'])
    bathrooms = st.number_input("Bathrooms (חדרי אמבטיה)", min_value=0.5, max_value=10.0, step=0.5, value=default_values['bathrooms'])
    bedrooms = st.number_input("Bedrooms (חדרי שינה)", min_value=0.0, max_value=10.0, step=1.0, value=default_values['bedrooms'])
    beds = st.number_input("Beds (מיטות)", min_value=0.0, max_value=20.0, step=1.0, value=default_values['beds'])

    property_type = st.selectbox("Property Type", property_types, index=property_types.index(default_values['property_type']) if default_values['property_type'] in property_types else 0)
    room_type = st.selectbox("Room Type", room_types, index=room_types.index(default_values['room_type']) if default_values['room_type'] in room_types else 0)
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods, index=neighbourhoods.index(default_values['neighbourhood_cleansed']) if default_values['neighbourhood_cleansed'] in neighbourhoods else 0)

    submitted = st.form_submit_button("חשב מחיר")

if submitted:
    # יצירת וקטור פיצ'רים מלא (כולל One Hot)
    input_dict = {
        'latitude': latitude,
        'longitude': longitude,
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': beds
    }

    # עמודות One Hot
    for p in property_types:
        input_dict[f'property_type_{p}'] = 1 if p == property_type else 0
    for r in room_types:
        input_dict[f'room_type_{r}'] = 1 if r == room_type else 0
    for n in neighbourhoods:
        input_dict[f'neighbourhood_cleansed_{n}'] = 1 if n == neighbourhood else 0

    # השלמת כל העמודות שהיו באימון
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    X_input = pd.DataFrame([input_dict], columns=feature_columns)

    # תחזית מחיר
    pred = model.predict(X_input)[0]
    st.success(f"מחיר צפוי ללילה: ${pred:,.0f}")
