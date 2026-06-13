import json
import random
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import optuna
import optuna.logging

from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from utils.utils import (
    connection,
    data_from_ticker,
    data_from_tpulse,
    data_from_macrofactors,
)

import utils.config_ml as config_ml

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

models = config_ml.MODELS


def pack_all_data_for_ml_models(ticker: str, left_date: str, right_date: str, conn):
    tpulse_data = data_from_tpulse(ticker, left_date, right_date, conn)

    ticker_data = data_from_ticker(ticker, left_date, right_date, conn)

    macrofactor_data = data_from_macrofactors(ticker, left_date, right_date, conn)

    data = tpulse_data.merge(macrofactor_data, how="left", on=["dt", "ticker"]).merge(
        ticker_data, how="left", on=["dt", "ticker"]
    )
    data = data[~data["target"].isnull()]

    return data


def sliding_windows_cross_validating(df, train_days, val_days, test_days, step):
    windows = []
    n = len(df)
    current_test_end = n
    while True:
        test_start = current_test_end - test_days
        if test_start < 0:
            break
        val_start = test_start - val_days
        if val_start < 0:
            break
        train_start = val_start - train_days
        if train_start < 0:
            break
        windows.append(
            {
                "train": df.iloc[train_start:val_start],
                "val": df.iloc[val_start:test_start],
                "test": df.iloc[test_start:current_test_end],
                "id": current_test_end,
                "dates": {
                    "train": (df["dt"].iloc[train_start], df["dt"].iloc[val_start - 1]),
                    "val": (df["dt"].iloc[val_start], df["dt"].iloc[test_start - 1]),
                    "test": (
                        df["dt"].iloc[test_start],
                        df["dt"].iloc[current_test_end - 1],
                    ),
                },
            }
        )
        current_test_end -= step
        if current_test_end < train_days + val_days + test_days:
            break
    return windows[::-1]


def select_top_features(model, feature_names, top_n):
    if hasattr(model, "coef_") and len(model.coef_) > 0:
        coefs = np.abs(model.coef_)
        top_indices = np.argsort(coefs)[-top_n:]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:]
    else:
        return feature_names[:top_n]
    return [feature_names[i] for i in top_indices]


def make_objective(model_name, X_train, y_train, X_val, y_val, metric_optuna):
    def objective(trial):
        model_cfg = models[model_name]
        params = {}
        if "optuna_objective" in model_cfg and callable(model_cfg["optuna_objective"]):
            params = model_cfg["optuna_objective"](trial)
        if model_name == "CatBoost":
            model = CatBoostRegressor(**params, verbose=False)
        else:
            model = model_cfg["model"]
            if params:
                model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric_optuna == "MAPE":
            return mean_absolute_percentage_error(y_val, y_pred)
        elif metric_optuna == "MAE":
            return mean_absolute_error(y_val, y_pred)

    return objective


### обучение обходится довольно дороговато (не очень критически, но желательно улучшить)
def train_val_test_ml_models(
    windows,
    ticker,
    left_date,
    right_date,
    train_period,
    val_period,
    test_period,
    step,
    metric_optuna,
    top_n_features,
):
    df_ml_db = pd.DataFrame(
        columns=[
            "dt",
            "ticker",
            "left_date",
            "right_date",
            "model_name",
            "model_value",
            "train_period",
            "val_period",
            "test_period",
            "step",
            "mape_mean",
            "mae_mean",
            "best_features",
        ]
    )
    df_ml = pd.DataFrame(columns=["test_period", "model_name", "mape", "mae"])
    for model in config_ml.MODELS:
        ml_info = []  # model, features, mape,
        for _, window in enumerate(windows):
            train, val, test = window["train"], window["val"], window["test"]
            feature_cols = [
                col for col in train.columns if col not in ["dt", "ticker", "target"]
            ]

            if model in ["LinearRegression", "DecisionTree", "GradientBoosting"]:
                train = train.fillna(0)
                val = val.fillna(0)
                test = test.fillna(0)

            X_train, y_train = train[feature_cols], train["target"]
            X_val, y_val = val[feature_cols], val["target"]
            X_test, y_test = test[feature_cols], test["target"]

            if model == "LinearRegression":
                best_params = {}
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                objective = make_objective(
                    model, X_train, y_train, X_val, y_val, metric_optuna
                )
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=4)
                best_params = study.best_params

            if model == "CatBoost":
                best_optuna_model = CatBoostRegressor(**best_params, verbose=False)
            else:
                best_optuna_model = config_ml.MODELS[model]["model"]
                best_optuna_model.set_params(**best_params)
            if model == "LinearRegression":
                best_optuna_model.fit(X_train_scaled, y_train)
            else:
                best_optuna_model.fit(X_train, y_train)

            selected_features = select_top_features(
                best_optuna_model, feature_cols, top_n_features
            )
            selected_idx = [
                feature_cols.index(feature) for feature in selected_features
            ]

            if model == "LinearRegression":
                best_optuna_model.fit(X_train_scaled[:, selected_idx], y_train)
                y_test_pred = best_optuna_model.predict(X_test_scaled[:, selected_idx])
            else:
                best_optuna_model.fit(X_train[selected_features], y_train)
                y_test_pred = best_optuna_model.predict(X_test[selected_features])
            test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            ml_info.append([best_optuna_model, selected_features, test_mape])

        best_model = sorted(ml_info, key=lambda x: x[-1])[0][0]
        best_features = set(ml_info[0][1])
        for feature_list in ml_info[1:]:
            best_features.intersection_update(feature_list[1])
        best_features = list(best_features)
        mape_mean, mae_mean = [], []

        for _, window in enumerate(windows):
            train, val, test = window["train"], window["val"], window["test"]
            X_train, y_train = train[best_features], train["target"]
            X_test, y_test = test[best_features], test["target"]
            best_model.fit(X_train, y_train)
            best_model_pred = best_model.predict(X_test)
            test_mape = mean_absolute_percentage_error(y_test, best_model_pred)
            test_mae = mean_absolute_error(y_test, best_model_pred)
            mape_mean.append(test_mape)
            mae_mean.append(test_mae)
            new_row = pd.Series(
                [
                    f"{window['dates']['test'][0]} - {window['dates']['test'][1]}",
                    model,
                    round(test_mape, 3),
                    round(test_mae, 3),
                ],
                index=df_ml.columns,
            )
            df_ml = pd.concat([df_ml, new_row.to_frame().T], ignore_index=True)

        new_row_db = pd.Series(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker,
                left_date,
                right_date,
                model,
                str(best_model),
                train_period,
                val_period,
                test_period,
                step,
                float(round(np.mean(mape_mean), 3)),
                float(round(np.mean(mae_mean), 3)),
                json.dumps(best_features),
            ],
            index=df_ml_db.columns,
        )
        df_ml_db = pd.concat([df_ml_db, new_row_db.to_frame().T], ignore_index=True)

    df_ml_db.to_sql("ml_models_data", con=connection(), if_exists="append", index=False)

    return df_ml.sort_values(by=["test_period", "mae"]), df_ml_db


def resulting(
    ticker,
    left_date,
    right_date,
    train_period,
    val_period,
    test_period,
    step,
    top_n_features,
    metric_optuna,
):
    start_time = time.time()
    data = pack_all_data_for_ml_models(ticker, left_date, right_date, connection())
    finish_time = time.time() - start_time
    print(f"pack {finish_time}")

    df = data.copy()
    df = df[~df["target"].isnull()]

    windows = sliding_windows_cross_validating(
        df, train_period, val_period, test_period, step
    )

    start_time = time.time()
    _, df_ml_db = train_val_test_ml_models(
        windows,
        ticker,
        left_date,
        right_date,
        train_period,
        val_period,
        test_period,
        step,
        metric_optuna,
        top_n_features,
    )
    finish_time = time.time() - start_time
    print(f"train,test,val {finish_time}")

    return df_ml_db.to_dict()


def make_prediction(ticker: str, interval: int):
    today_date = datetime.today().date()
    first_date = today_date - timedelta(days=interval)
    data = pd.read_sql_query(
        f"""
            SELECT ticker, datetime, close
            FROM candles
            WHERE 1=1
            AND ticker = '{ticker}'
            AND datetime >= '{first_date}'
        """,
        connection(),
    )

    min_price = min(data["close"]) * 0.95
    max_price = max(data["close"]) * 1.05
    predicted_price = [
        round(random.uniform(min_price, max_price), 1) for _ in range(interval)
    ]
    future_dates = [
        (max(data["datetime"]) + timedelta(days=i)).date()
        for i in range(1, interval + 1)
    ]
    result_df = pd.DataFrame([future_dates, predicted_price]).T
    result_df.columns = ["dt", "price"]

    return result_df.to_dict()
