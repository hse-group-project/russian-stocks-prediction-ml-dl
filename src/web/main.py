import streamlit as st
import requests


def predict():
    data = requests.get("http://localhost:8000/")
    res = data.json()
    return res

message = predict()
    
st.write(message)
