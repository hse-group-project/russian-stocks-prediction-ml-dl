#!/usr/bin/env python3
import os
import random
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
from botocore.exceptions import NoCredentialsError, EndpointConnectionError

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import (
    connection,
    data_from_macrofactors,
    data_from_ticker,
    data_from_tpulse,
)


# ============================================================
# 1. Вспомогательные функции
# ============================================================
def test_s3_connection(cfg: DictConfig):
    """Проверка доступности MinIO/S3."""
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=cfg.env.mlflow_s3_endpoint_url,
            aws_access_key_id=cfg.env.aws_access_key_id,
            aws_secret_access_key=cfg.env.aws_secret_access_key,
        )
        buckets = s3.list_buckets()
        print(f"S3 connected. Buckets: {[b['Name'] for b in buckets['Buckets']]}")
        return True
    except NoCredentialsError:
        print("No credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    except EndpointConnectionError as e:
        print(f"Cannot connect to S3 endpoint: {e}")
        return False


def set_seed(seed: int):
    """Фиксация seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)


def sliding_windows_cross_validation(df, train_days, val_days, test_days, step):
    """Генерация скользящих окон (train/val/test)."""
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


# ============================================================
# 2. Логирование артефактов
# ============================================================
def log_regression_artifacts(run_dir, y_true, y_pred, feature_cols=None):
    """Графики и примеры для регрессии."""
    residuals = y_true - y_pred

    plt.figure(figsize=(5, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.savefig(
        os.path.join(run_dir, "pred_vs_actual.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    mlflow.log_artifact(
        os.path.join(run_dir, "pred_vs_actual.png"), artifact_path="plots"
    )

    plt.figure(figsize=(5, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(0, color="r", linestyle="--")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.savefig(os.path.join(run_dir, "residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(os.path.join(run_dir, "residuals.png"), artifact_path="plots")

    sample = pd.DataFrame(
        {
            "actual": y_true.iloc[:20].values,
            "predicted": y_pred[:20],
            "error": residuals[:20],
        }
    )
    sample.to_csv(os.path.join(run_dir, "sample_predictions.csv"), index=False)
    mlflow.log_artifact(
        os.path.join(run_dir, "sample_predictions.csv"), artifact_path="data"
    )


def analyze_regression_errors(y_true, y_pred, run_dir, top_n=20):
    """Анализ 10-20 самых больших ошибок."""
    abs_errors = np.abs(y_true - y_pred)
    error_idx = np.argsort(abs_errors)[-top_n:][::-1]

    errors = pd.DataFrame(
        {
            "index": error_idx,
            "actual": y_true.iloc[error_idx].values,
            "predicted": y_pred[error_idx],
            "error": y_true.iloc[error_idx].values - y_pred[error_idx],
            "error_type": np.where(
                y_pred[error_idx] > y_true.iloc[error_idx].values,
                "OVERPREDICT",
                "UNDERPREDICT",
            ),
        }
    )

    err_path = os.path.join(run_dir, "error_analysis.csv")
    errors.to_csv(err_path, index=False, encoding="utf-8-sig")
    mlflow.log_artifact(err_path, artifact_path="analysis")

    over_pct = (errors["error_type"] == "OVERPREDICT").mean() * 100
    under_pct = 100 - over_pct
    mean_abs_err = errors["error"].abs().mean()

    report_md = f"""# Анализ ошибок модели
                    ## Общая статистика
                    - Проанализировано ошибок: {top_n}
                    - Доля OVERPREDICT (завышение): {over_pct:.1f}%
                    - Доля UNDERPREDICT (занижение): {under_pct:.1f}%
                    - Средняя абсолютная ошибка в топ-{top_n}: {mean_abs_err:.4f}
                """
    report_path = os.path.join(run_dir, "error_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    mlflow.log_artifact(report_path, artifact_path="analysis")

    print(
        f"Топ-{top_n} ошибок сохранены. OVER: {over_pct:.1f}%, UNDER: {under_pct:.1f}%"
    )
    return errors


def check_robustness(model, X_test, y_test, noise_level=0.01):
    """Устойчивость к шуму (1% от std каждого признака)."""
    X_noisy = X_test.copy()
    for col in X_noisy.select_dtypes(include=[np.number]).columns:
        std = X_noisy[col].std()
        if std > 0:
            X_noisy[col] += np.random.normal(0, noise_level * std, len(X_noisy))

    y_orig = model.predict(X_test)
    y_noisy = model.predict(X_noisy)

    mape_orig = mean_absolute_percentage_error(y_test, y_orig)
    mape_noisy = mean_absolute_percentage_error(y_test, y_noisy)

    stability = float(np.abs(mape_noisy - mape_orig) / (mape_orig + 1e-8))
    mlflow.log_metric("robustness/mape_deviation_1pct_noise", stability)
    print(f"Robustness: отклонение MAPE при шуме 1% = {stability * 100:.2f}%")
    return stability


def save_data_snapshot(top_features: list, run_dir: str, metadata: dict = None):
    """Сохраняет снапшот данных (список признаков + метаданные) в pickle."""
    snapshot = {
        "top_features": top_features,
        "n_features": len(top_features),
        "created_at": datetime.now().isoformat(),
        "model_type": "xgb_regressor_fixed_params",
    }
    if metadata:
        snapshot.update(metadata)

    snapshot_path = os.path.join(run_dir, "data_snapshot.pkl")
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)

    mlflow.log_artifact(snapshot_path, artifact_path="data")
    print(f"Снапшот данных сохранён: {len(top_features)} признаков")
    return snapshot_path


def pack_all_data_for_ml_models(ticker: str, left_date: str, right_date: str, conn):
    tpulse_data = data_from_tpulse(ticker, left_date, right_date, conn)

    ticker_data = data_from_ticker(ticker, left_date, right_date, conn)

    macrofactor_data = data_from_macrofactors(ticker, left_date, right_date, conn)

    data = tpulse_data.merge(macrofactor_data, how="left", on=["dt", "ticker"]).merge(
        ticker_data, how="left", on=["dt", "ticker"]
    )
    data = data[~data["target"].isnull()]

    return data


# ============================================================
# 3. Основная функция обучения
# ============================================================
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # --- Применяем переменные окружения из конфига ---
    os.environ["MLFLOW_TRACKING_URI"] = cfg.env.mlflow_tracking_uri
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.env.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.env.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.env.mlflow_s3_endpoint_url

    # --- Загружаем параметры из конфига ---
    companies = pd.read_sql("SELECT * FROM companies", connection())
    tickers = companies["ticker"]
    left_date = list(cfg.data.left_dates)
    right_date = list(cfg.data.right_dates)
    train_period = cfg.data.train_period
    val_period = cfg.data.val_period
    test_period = cfg.data.test_period
    step = cfg.data.step
    top_n_features = cfg.data.top_n_features

    # Гиперпараметры модели как dict
    xgb_params = OmegaConf.to_container(cfg.model, resolve=True)

    CONFIG = {
        "experiment_name": cfg.experiment.name,
        "run_name": cfg.experiment.run_name,
        "tracking_uri": cfg.env.mlflow_tracking_uri,
        "tickers": tickers,
        "top_n_features": top_n_features,
        "seed": cfg.seed,
        "data_description": (
            "Скользящие окна по тикерам. Целевая переменная: target (регрессия). "
            "Модель выбрана по минимальному MAPE на валидации среди всех окон и тикеров."
        ),
    }

    set_seed(CONFIG["seed"])
    mlflow.set_tracking_uri(CONFIG["tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])

    run_dir = cfg.artifacts.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # --- Сохраняем сам конфиг как артефакт (бонус для воспроизводимости) ---
    cfg_path = os.path.join(run_dir, "hydra_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # --- Трекеры лучшей модели ---
    best_model = None
    best_top_feats = None
    best_val_mape = float("inf")
    best_window_info = None
    best_X_te = None
    best_y_te = None

    # --- Агрегированные метрики ---
    mape_list, mae_list = [], []
    baseline_mape_list, baseline_mae_list = [], []

    with mlflow.start_run(run_name=CONFIG["run_name"]) as run:
        # 1. Логирование параметров
        mlflow.log_params(xgb_params)
        mlflow.log_param("seed", CONFIG["seed"])
        mlflow.log_param("top_n_features", CONFIG["top_n_features"])
        mlflow.log_param("data_description", CONFIG["data_description"])
        mlflow.log_param("model_selection_criterion", "min MAPE on validation set")
        mlflow.log_artifact(cfg_path, artifact_path="config")

        # 2. Цикл обучения по тикерам и окнам
        for ticker in CONFIG["tickers"]:
            print(f"\n{'=' * 50}\nТикер: {ticker}")

            for left, right in zip(left_date, right_date):
                if os.path.exists(
                    f"{cfg.data.processed_path}/{ticker}/{left}_{right}.pkl.gz"
                ):
                    df = pd.read_pickle(
                        f"{cfg.data.processed_path}/{ticker}/{left}_{right}.pkl.gz"
                    )

                else:
                    data = pack_all_data_for_ml_models(
                        ticker, left, right, connection()
                    )
                    df = data[~data["target"].isnull()].sort_values("dt")

                windows = sliding_windows_cross_validation(
                    df, train_period, val_period, test_period, step
                )

                for idx, window in enumerate(windows):
                    train, val, test = window["train"], window["val"], window["test"]
                    exclude = ["dt", "ticker", "ticker_x", "ticker_y", "target"]
                    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
                    feat_cols = [c for c in num_cols if c not in exclude]

                    train, val, test = [d.fillna(0) for d in [train, val, test]]
                    X_tr, y_tr = train[feat_cols], train["target"]
                    X_val, y_val = val[feat_cols], val["target"]
                    X_te, y_te = test[feat_cols], test["target"]

                    # Обучение для отбора признаков
                    model = xgb.XGBRegressor(**xgb_params)
                    model.fit(X_tr, y_tr)

                    # Выбор топ-фич
                    imp = model.feature_importances_
                    top_feats = [
                        f
                        for f, _ in sorted(
                            zip(feat_cols, imp), key=lambda x: x[1], reverse=True
                        )[: CONFIG["top_n_features"]]
                    ]

                    # Обучение финальной модели на топ-фичах
                    final_model = xgb.XGBRegressor(**xgb_params)
                    final_model.fit(X_tr[top_feats], y_tr)

                    # Оценка на ВАЛИДАЦИИ для выбора лучшей модели
                    y_val_pred = final_model.predict(X_val[top_feats])
                    val_mape = mean_absolute_percentage_error(y_val, y_val_pred)

                    if val_mape < best_val_mape:
                        best_val_mape = val_mape
                        best_model = final_model
                        best_top_feats = top_feats
                        best_window_info = {
                            "ticker": ticker,
                            "window_idx": idx,
                            "dates": window["dates"],
                            "val_mape": val_mape,
                        }
                        best_X_te = X_te
                        best_y_te = y_te
                        print(f"Новая лучшая модель: val_MAPE={val_mape:.5f}")

                    # Предсказание на тесте
                    y_pred = final_model.predict(X_te[top_feats])
                    mape = mean_absolute_percentage_error(y_te, y_pred)
                    mae = mean_absolute_error(y_te, y_pred)

                    mape_list.append(mape)
                    mae_list.append(mae)

                    mlflow.log_metric(f"{ticker}/window_{idx}_mape", mape)
                    mlflow.log_metric(f"{ticker}/window_{idx}_mae", mae)
                    mlflow.log_metric(f"{ticker}/window_{idx}_val_mape", val_mape)

                    # Baseline
                    baseline = DummyRegressor(strategy="mean")
                    baseline.fit(X_tr, y_tr)
                    y_bl = baseline.predict(X_te)
                    bl_mape = mean_absolute_percentage_error(y_te, y_bl)
                    bl_mae = mean_absolute_error(y_te, y_bl)

                    baseline_mape_list.append(bl_mape)
                    baseline_mae_list.append(bl_mae)

                    mlflow.log_metric(f"{ticker}/window_{idx}_baseline_mape", bl_mape)
                    mlflow.log_metric(f"{ticker}/window_{idx}_baseline_mae", bl_mae)

                    print(
                        f"  Окно {idx}: MAPE={mape:.5f} (Δ vs baseline: {bl_mape - mape:+.5f})"
                    )

        # === Проверка, что лучшая модель найдена ===
        if best_model is None:
            raise RuntimeError("Не удалось найти ни одной модели для сохранения")

        print(f"\n{'=' * 50}")
        print("ЛУЧШАЯ МОДЕЛЬ:")
        print(f"Тикер: {best_window_info['ticker']}")
        print(f"Окно: {best_window_info['window_idx']}")
        print(f"Val MAPE: {best_window_info['val_mape']:.5f}")
        print(f"Период теста: {best_window_info['dates']['test']}")

        # === Агрегированные метрики ===
        mean_mape = np.mean(mape_list)
        mean_mae = np.mean(mae_list)
        mean_bl_mape = np.mean(baseline_mape_list)
        mean_bl_mae = np.mean(baseline_mae_list)

        mlflow.log_metric("mean_MAPE", mean_mape)
        mlflow.log_metric("mean_MAE", mean_mae)
        mlflow.log_metric("baseline/mean_MAPE", mean_bl_mape)
        mlflow.log_metric("baseline/mean_MAE", mean_bl_mae)
        mlflow.log_metric("best/val_MAPE", best_val_mape)

        print("\nИтог:")
        print(f"Model   — MAPE: {mean_mape:.5f}, MAE: {mean_mae:.3f}")
        print(f"Baseline— MAPE: {mean_bl_mape:.5f}, MAE: {mean_bl_mae:.3f}")
        print(
            f"Δ Improvement: {mean_bl_mape - mean_mape:+.5f} MAPE, {mean_bl_mae - mean_mae:+.3f} MAE"
        )

        # === Артефакты на лучшем окне ===
        y_final_pred = best_model.predict(best_X_te[best_top_feats])
        log_regression_artifacts(run_dir, best_y_te, y_final_pred, best_top_feats)
        analyze_regression_errors(
            best_y_te, y_final_pred, run_dir, top_n=cfg.artifacts.top_n_errors
        )

        # === Robustness check ===
        check_robustness(
            best_model,
            best_X_te[best_top_feats],
            best_y_te,
            noise_level=cfg.artifacts.noise_level,
        )

        # === Сохранение снапшота данных ===
        save_data_snapshot(
            top_features=best_top_feats,
            run_dir=run_dir,
            metadata={
                "experiment_name": CONFIG["experiment_name"],
                "run_name": CONFIG["run_name"],
                "seed": CONFIG["seed"],
                "top_n_features": CONFIG["top_n_features"],
                "data_description": CONFIG["data_description"],
                "mean_MAPE": mean_mape,
                "mean_MAE": mean_mae,
                "best_ticker": best_window_info["ticker"],
                "best_window_idx": best_window_info["window_idx"],
                "best_val_MAPE": best_val_mape,
                "train_dates": best_window_info["dates"]["train"],
                "test_dates": best_window_info["dates"]["test"],
            },
        )

        # === Сохранение модели + теги PRD ===
        mlflow.set_tag("stage", "PRD")
        mlflow.set_tag("model_type", "xgb_regressor_fixed_params")
        mlflow.set_tag("git_commit", "main")
        mlflow.set_tag(
            "description",
            f"Final XGBoost model. Best on ticker={best_window_info['ticker']}, "
            f"window={best_window_info['window_idx']}, val_MAPE={best_val_mape:.5f}",
        )

        mlflow.xgboost.log_model(
            best_model,
            name="model",
            signature=mlflow.models.infer_signature(
                best_X_te[best_top_feats], best_y_te
            ),
        )

        print(f"\nМодель залогирована. Run ID: {run.info.run_id}")
        print("Тег stage=PRD установлен")


# ============================================================
# 4. Точка входа
# ============================================================
if __name__ == "__main__":
    main()
