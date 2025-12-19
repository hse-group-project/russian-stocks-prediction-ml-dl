import streamlit as st
import requests
from datetime import date
import utils.config_ml as config_ml
from src.web.utils.utils import graphic

TICKER_DEFAULT = config_ml.TICKER
LEFT_DATE_DEFAULT = config_ml.LEFT_DATE
RIGHT_DATE_DEFAULT = config_ml.RIGHT_DATE
TRAIN_PERIOD_DEFAULT = config_ml.TRAIN_PERIOD
VAL_PERIOD_DEFAULT = config_ml.VAL_PERIOD
TEST_PERIOD_DEFAULT = config_ml.TEST_PERIOD
STEP_DEFAULT = config_ml.STEP
# N_TRIALS_DEFAULT = config_ml.N_TRIALS
METRIC_OPTUNA_DEFAULT = config_ml.METRIC_OPTUNA
TOP_N_FEATURES_DEFAULT = config_ml.TOP_N_FEATURES

st.set_page_config(page_title="Main", layout="centered")


def predict_data(
    ticker: str,
    left_date: str,
    right_date: str,
    train_period: int,
    val_period: int,
    test_period: int,
    step: int,
    top_n_features: int,
    metric_optuna: str,
):
    try:
        json_params = {
            "ticker": ticker,
            "left_date": left_date,
            "right_date": right_date,
            "train_period": train_period,
            "val_period": val_period,
            "test_period": test_period,
            "step": step,
            "top_n_features": top_n_features,
            "metric_optuna": metric_optuna,
        }

        response = requests.post("http://localhost:8000/api/forward", json=json_params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error predict data: {e}")
        return []


def fetch_fake_predict(ticker, interval):
    try:
        json_params = {
            "ticker": ticker,
            "interval": interval,
        }
        response = requests.post(
            "http://localhost:8000/api/fake_predict", json=json_params
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


def fetch_companies():
    try:
        response = requests.get("http://localhost:8000/api/companies")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


def fetch_candles(ticker: str, left_date: str, right_date: str):
    try:
        params = {"ticker": ticker, "left_date": left_date, "right_date": right_date}
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
if "predict_data" not in st.session_state:
    st.session_state.predict_data = []
if "predict_interval" not in st.session_state:
    st.session_state.predict_interval = []
if "companies" not in st.session_state:
    st.session_state.companies = []

st.session_state.companies = fetch_companies()

with st.form(key="input_form"):
    st.subheader("Fetch Candle Data")
    ticker_input = st.selectbox("Ticker", options=st.session_state.companies)

    try:
        left_default = date.fromisoformat(LEFT_DATE_DEFAULT)
        right_default = date.fromisoformat(RIGHT_DATE_DEFAULT)
    except Exception:
        left_default = date.today()
        right_default = date.today()

    left_date_input = st.date_input("Start Date", value=left_default)
    right_date_input = st.date_input("End Date", value=right_default)
    interval = st.number_input(label="Interval", value=14)
    train_period = st.number_input(label="train_period", value=TRAIN_PERIOD_DEFAULT)
    val_period = st.number_input(label="val_period", value=VAL_PERIOD_DEFAULT)
    test_period = st.number_input(label="test_period", value=TEST_PERIOD_DEFAULT)
    step = st.number_input(label="step", value=STEP_DEFAULT)
    # st.number_input(label="n_trials", value=N_TRIALS_DEFAULT, key="n_trials")
    top_n_features = st.number_input(
        label="top_n_features", value=TOP_N_FEATURES_DEFAULT
    )
    metric_optuna = st.selectbox(label="metric_optuna", options=["MAPE", "MAE"])

    submit = st.form_submit_button("Predict Data")

if submit:
    left_str = left_date_input.isoformat()
    right_str = right_date_input.isoformat()

    st.session_state.candles_data = fetch_candles(ticker_input, left_str, right_str)

    st.session_state.predict_data = predict_data(
        ticker_input,
        left_str,
        right_str,
        train_period,
        val_period,
        test_period,
        step,
        top_n_features,
        metric_optuna,
    )

    st.session_state.predict_interval = fetch_fake_predict(ticker_input, interval)

if st.session_state.candles_data:
    st.divider()
    try:
        fig = graphic(st.session_state.candles_data)
        st.markdown("#### Candlestick Chart", text_alignment="center")
        st.plotly_chart(fig, width="content")
    except Exception as e:
        st.error(f"Error rendering graph: {e}")
else:
    st.info("Enter parameters and click 'Fetch Data' to load candlestick chart.")

if st.session_state.predict_data:
    st.divider()
    st.markdown("#### Results of predict", text_alignment="center")
    st.dataframe(st.session_state.predict_data)

if st.session_state.predict_interval:
    st.divider()
    st.markdown("#### Fake predict", text_alignment="center")
    st.dataframe(st.session_state.predict_interval)
