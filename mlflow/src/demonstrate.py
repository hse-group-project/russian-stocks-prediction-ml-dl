#!/usr/bin/env python3
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Применяем переменные окружения
    os.environ["MLFLOW_TRACKING_URI"] = cfg.env.mlflow_tracking_uri
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.env.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.env.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.env.mlflow_s3_endpoint_url

    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ФИНАЛЬНОЙ МОДЕЛИ (PRD)")
    print("=" * 60)

    experiment_name = cfg.experiment.name

    # 1. Загрузка модели
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Эксперимент '{experiment_name}' не найден в MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'PRD'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise ValueError("Модель с тегом stage='PRD' не найдена.")

    run = runs[0]
    run_id = run.info.run_id
    print(f"Найден PRD-run: {run_id}")
    print(f"   Описание: {run.data.tags.get('description', '—')}")

    # Загрузка модели
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Модель загружена из {model_uri}")

    # Загрузка списка признаков
    top_features = None
    try:
        local_path = client.download_artifacts(
            run_id=run_id, path="data/data_snapshot.pkl"
        )
        with open(local_path, "rb") as f:
            snapshot = pickle.load(f)
        top_features = snapshot["top_features"]
        print(f"Список признаков загружен: {len(top_features)} фич")
    except Exception as e:
        print(f"Не удалось загрузить data_snapshot.pkl: {e}")
        sys.exit(1)

    # Метрики
    metrics = run.data.metrics
    print("\nМетрики PRD-модели:")
    for key in [
        "mean_MAPE",
        "mean_MAE",
        "baseline/mean_MAPE",
        "baseline/mean_MAE",
        "best/val_MAPE",
    ]:
        if key in metrics:
            print(f"   {key}: {metrics[key]:.5f}")

    # 2. Подготовка данных (используем параметры из конфига)
    n_samples = cfg.get("inference", {}).get("n_samples", 5)
    seed = cfg.get("inference", {}).get("seed", 123)
    input_file = cfg.get("inference", {}).get("input_data", None)

    if input_file and Path(input_file).exists():
        print(f"\nЗагружаем данные из {input_file}")
        data = pd.read_csv(input_file)
        missing = set(top_features) - set(data.columns)
        if missing:
            raise ValueError(f"В файле отсутствуют колонки: {missing}")
        data = data[top_features]
    else:
        print(f"\nГенерируем синтетические данные (n={n_samples}, seed={seed})")
        np.random.seed(seed)
        data = pd.DataFrame(
            np.random.randn(n_samples, len(top_features)),
            columns=top_features,
        )

    print(f"\nВходные данные: shape={data.shape}")

    # 3. Инференс
    predictions = model.predict(data)
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА")
    print("=" * 60)
    results = pd.DataFrame(
        {
            "sample_idx": range(1, len(predictions) + 1),
            "prediction": predictions,
        }
    )
    print(results.to_string(index=False))

    print("\nСтатистика предсказаний:")
    print(f"   Среднее: {predictions.mean():.4f}")
    print(f"   Медиана: {np.median(predictions):.4f}")
    print(f"   Мин/Макс: {predictions.min():.4f} / {predictions.max():.4f}")

    if "mean_MAPE" in metrics:
        print(f"\nОжидаемая ошибка модели (MAPE): {metrics['mean_MAPE'] * 100:.2f}%")

    # 4. Robustness
    if not cfg.get("inference", {}).get("no_robustness", False):
        print("\n" + "=" * 60)
        print("ПРОВЕРКА УСТОЙЧИВОСТИ (ROBUSTNESS)")
        print("=" * 60)

        y_orig = model.predict(data)
        data_noisy = data.copy()
        noise_level = cfg.artifacts.noise_level
        for col in data_noisy.columns:
            std = data_noisy[col].std()
            if std > 0:
                data_noisy[col] += np.random.normal(
                    0, noise_level * std, len(data_noisy)
                )
        y_noisy = model.predict(data_noisy)

        deviation = np.abs(y_noisy - y_orig)
        mean_dev = deviation.mean()
        max_dev = deviation.max()
        rel_dev = (deviation / (np.abs(y_orig) + 1e-8)).mean() * 100

        print(f"Шум: {noise_level * 100:.1f}% от std каждого признака")
        print(f"Среднее абсолютное отклонение: {mean_dev:.4f}")
        print(f"Максимальное отклонение: {max_dev:.4f}")
        print(f"Среднее относительное отклонение: {rel_dev:.2f}%")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Samples: {len(predictions)}")
    print(f"Features: {len(top_features)}")
    if "mean_MAPE" in metrics:
        print(f"Model MAPE: {metrics['mean_MAPE'] * 100:.2f}%")
    print("\nДемонстрация завершена успешно.")


if __name__ == "__main__":
    main()
