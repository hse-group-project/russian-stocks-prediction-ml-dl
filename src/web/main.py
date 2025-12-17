import streamlit as st
import requests
from datetime import date
import utils.config_ml as config_ml
from src.web.utils.utils import graphic

TICKER_DEFAULT = config_ml.TICKER
LEFT_DATE_DEFAULT = config_ml.LEFT_DATE
RIGHT_DATE_DEFAULT = config_ml.RIGHT_DATE

st.set_page_config(page_title="Main", layout="centered")


def fetch_candles(ticker: str, left_date: str, right_date: str):
    try:
        params = {"left_date": left_date, "right_date": right_date, "ticker": ticker}
        response = requests.get("http://localhost:8000/api/candles", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


def fetch_indices(ticker: str, left_date: str, right_date: str):
    try:
        params = {"left_date": left_date, "right_date": right_date, "ticker": ticker}
        response = requests.get("http://localhost:8000/api/indices", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


if "candles_data" not in st.session_state:
    st.session_state.candles_data = []

with st.form(key="input_form"):
    st.subheader("Fetch Candle Data")
    ticker_input = st.text_input("Ticker", value=TICKER_DEFAULT)

    try:
        left_default = date.fromisoformat(LEFT_DATE_DEFAULT)
        right_default = date.fromisoformat(RIGHT_DATE_DEFAULT)
    except Exception:
        left_default = date.today()
        right_default = date.today()

    left_date_input = st.date_input("Start Date", value=left_default)
    right_date_input = st.date_input("End Date", value=right_default)

    submit = st.form_submit_button("Fetch Data")

if submit:
    left_str = left_date_input.isoformat()
    right_str = right_date_input.isoformat()

    st.session_state.candles_data = fetch_candles(ticker_input, left_str, right_str)

if st.session_state.candles_data:
    try:
        fig = graphic(st.session_state.candles_data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering graph: {e}")
else:
    st.info("Enter parameters and click 'Fetch Data' to load candlestick chart.")
