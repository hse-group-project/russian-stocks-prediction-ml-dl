import requests
import streamlit as st


st.set_page_config(page_title="Tools", layout="centered")

st.subheader("Tools")


def get_history():
    try:
        response = requests.get("http://localhost:8000/api/history")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


def delete_history():
    try:
        response = requests.delete("http://localhost:8000/api/history")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


def get_stats():
    try:
        response = requests.get("http://localhost:8000/api/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


if "history_data" not in st.session_state:
    st.session_state.history_data = []
if "stats_data" not in st.session_state:
    st.session_state.stats_data = []


get_history_bn = st.button("Get history requests")
del_history_bn = st.button("Delete history requests")

get_stats_bn = st.button("Get requests stats")

if get_history_bn:
    st.session_state.history_data = get_history()

if del_history_bn:
    delete_history()

if get_stats_bn:
    st.session_state.stats_data = get_stats()
