import streamlit as st
import utils.config_ml as config_ml


st.set_page_config(page_title="Config", layout="centered")

st.subheader("Config")

TRAIN_PERIOD_DEFAULT = config_ml.TRAIN_PERIOD
VAL_PERIOD_DEFAULT = config_ml.VAL_PERIOD
TEST_PERIOD_DEFAULT = config_ml.TEST_PERIOD
STEP_DEFAULT = config_ml.STEP
N_TRIALS_DEFAULT = config_ml.N_TRIALS
METRIC_OPTUNA_DEFAULT = config_ml.METRIC_OPTUNA
TOP_N_FEATURES_DEFAULT = config_ml.TOP_N_FEATURES

train_period = st.number_input(label="train_period", value=TRAIN_PERIOD_DEFAULT)
val_period = st.number_input(label="val_period", value=VAL_PERIOD_DEFAULT)
test_period = st.number_input(label="test_period", value=TEST_PERIOD_DEFAULT)
step = st.number_input(label="step", value=STEP_DEFAULT)
n_trials = st.number_input(label="n_trials", value=N_TRIALS_DEFAULT)
metric_optuna = st.selectbox(label="metric_optuna", options=["MAPE", "MAE"])
top_n_features = st.number_input(label="top_n_features", value=TOP_N_FEATURES_DEFAULT)
