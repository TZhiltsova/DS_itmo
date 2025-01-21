import os

import pandas as pd
import streamlit as st

from ml import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Flat price",
)

model_path = 'model.pkl'

rooms = st.sidebar.selectbox("Сколько комнат в квартире (0, если студия?", (0, 1, 2, 3))
lon = st.sidebar.number_input(
    "Долгота координаты квартиры?"
)
lat = st.sidebar.number_input(
    "Широта"
)

# create input DataFrame
inputDF = pd.DataFrame(
    {
        "lat": lat,
        "lon": lon,
        "rooms": rooms,
    },
    index=[0],
)

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model(model_path)

preds = model.predict(inputDF)

if st.sidebar.button('Узнать цену'):
    st.write(f"Цена квартиры: {preds}%")