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

# predict button

if st.button("Predict HDB Price: "):
    input_data = {
        "town": town_selected,
        "flat_type": flat_type_selected,
        "storey_range": storey_range_selected,
        "floor_area": floor_area_selected
    }

    df_input = pd.DataFrame([input_data])

    df_input = df_input.reindex(columns= model.feature_names_in_, fill_value=0)
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted resale price: ${y_unseen_pred:, .2f}")

st.markdown(
    unsafe_allow_html= True
)