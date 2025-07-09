import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("resale-flat.pkl")

st.title("HDB Resale flat prediction")

towns = ["Tampines", "Bedok", "Punggol"]
flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM"]
storey_ranges = ["01 TO 03", "04 TO 06", "07 TO 09"]

# calling input/form elements

# selectbox = dropdown box
# slider = slider element
# try input box?
town_selected = st.selectbox("Select Towns:", towns)
flat_type_selected = st.selectbox("Select Flat Type: ", flat_types)
storey_range_selected = st.selectbox("Select the storey range: ", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (spm)", min_value=10, max_value=200, value=70)

if st.button("Predict Price"):
    # Create a dictionary for the input features
    input_data = {
        'town': selected_town,
        'flat_type': selected_flat_type,
        'storey_range': selected_storey_range,
        'floor_area_sqm': selected_floor_area_sqm
    }
    
    # Convert input data to a DataFrame and one-hot encode
    input_df= pd.DataFrame({'town': [selected_town],
                            'flat_type': [selected_flat_type],
                            'storey_range': [selected_storey_range],
                            'floor_area_sqm': [selected_floor_area_sqm]})
    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)


    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Resale Price: ${prediction:,.2f}")

st.markdown(
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>,
    unsafe_allow_html=True
)
