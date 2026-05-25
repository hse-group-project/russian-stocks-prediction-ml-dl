import os
from dotenv import load_dotenv

from joblib import Parallel, delayed
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from datetime import datetime, timedelta

import re
import ast
import spacy
from collections import Counter
from nltk.corpus import stopwords

from .config_ml import TRAIN_PERIOD, VAL_PERIOD, TEST_PERIOD, TOP_100_WORDS, INDICES

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

load_dotenv()

# Глобальные переменные
shift_days = max(TRAIN_PERIOD, VAL_PERIOD, TEST_PERIOD)
top_words_list = TOP_100_WORDS
indices_list = INDICES
nlp = spacy.load("ru_core_news_sm")
russian_stopwords = set(
    list(set(stopwords.words("russian"))) + ["это", "р", "г", "руб", "шт"]
)


# Функция подключения к БД
def connection():
    db_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }
    conn_str = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(conn_str)


# Функции для обработки текстов
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)  # только буквы
    text = re.sub(r"\d+", " ", text)  # убираем цифры
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_reactions(reactions_str):
    try:
        reactions = ast.literal_eval(reactions_str)
    except Exception:
        reactions = []
    reaction_dict = {
        f"reaction_{r['type']}": r["count"]
        for r in reactions
        if "type" in r and "count" in r
    }
    for r in ["buy-up", "rocket", "not-convinced", "get-rid", "like", "dislike"]:
        reaction_dict.setdefault(f"reaction_{r}", 0)
    reaction_dict["total_reactions"] = sum(reaction_dict.values())
    return pd.Series(reaction_dict)


def lemmatize_no_stop(text):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.text not in russian_stopwords and t.is_alpha]


def lemmatize_no_stop_batch(texts, batch_size):
    """Лемматизация батчами для экономии памяти"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        docs = list(nlp.pipe(batch))
        lemmas = [
            [t.lemma_ for t in doc if t.text not in russian_stopwords and t.is_alpha]
            for doc in docs
        ]
        results.extend(lemmas)
    return results


def top_words_stats(lemmas):
    total_words = len(lemmas)
    counts = Counter(lemmas)
    stats = {}
    for word in top_words_list:
        stats[f"tfidf_{word}"] = (
            counts.get(word, 0) / total_words if total_words > 0 else 0
        )
    stats["top_words_pct"] = sum([stats[f"tfidf_{w}"] for w in top_words_list])
    return pd.Series(stats)


def lemmatize_parallel(texts, n_workers=4, batch_size=64):
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def process_chunk(chunk):
        docs = list(nlp.pipe(chunk))
        return [
            [t.lemma_ for t in doc if t.text not in russian_stopwords and t.is_alpha]
            for doc in docs
        ]

    with Parallel(n_jobs=n_workers, backend="threading") as parallel:
        results = parallel(delayed(process_chunk)(chunk) for chunk in chunks)

    return [lemma for chunk in results for lemma in chunk]


# Функция feature engineering с Т-Пульс
def data_from_tpulse(
    ticker: str, left_date: str, right_date: str, conn
) -> pd.DataFrame:
    query = """
        SELECT * FROM t_pulse_data
        WHERE ticker = %s
        AND inserted >= %s AND inserted <= %s
        ORDER BY inserted ASC
    """
    t_pulse_data = pd.read_sql(query, conn, params=(ticker, left_date, right_date))

    if t_pulse_data.empty:
        return pd.DataFrame(columns=["dt", "ticker"])

    t_pulse_features = t_pulse_data.drop(columns=["id"]).rename(
        columns={"inserted": "dt"}
    )
    t_pulse_features["dt"] = pd.to_datetime(
        t_pulse_features["dt"], errors="coerce"
    ).dt.date

    t_pulse_features["clean_text"] = t_pulse_features["text"].apply(clean_text)
    t_pulse_features["lemmas"] = lemmatize_parallel(
        t_pulse_features["clean_text"].tolist()
    )

    t_pulse_features["num_words"] = t_pulse_features["lemmas"].str.len()
    t_pulse_features["num_chars"] = t_pulse_features["clean_text"].str.len()

    reactions_df = t_pulse_features["reactions_counters"].apply(parse_reactions)
    t_pulse_features = pd.concat([t_pulse_features, reactions_df], axis=1)

    t_pulse_features["commentscount"] = pd.to_numeric(
        t_pulse_features["commentscount"], errors="coerce"
    ).fillna(0)
    t_pulse_features["comment_strength"] = np.log(
        np.exp(1) + t_pulse_features["commentscount"]
    )

    top_words_df = t_pulse_features["lemmas"].apply(top_words_stats)
    t_pulse_features = pd.concat([t_pulse_features, top_words_df], axis=1)

    t_pulse_features = t_pulse_features.drop(
        columns=["text", "clean_text", "lemmas", "reactions_counters"]
    )

    agg_cols = t_pulse_features.drop(columns=["dt", "ticker"]).columns
    t_pulse_features_gr = t_pulse_features.groupby(["dt", "ticker"], as_index=False)[
        agg_cols
    ].agg(["sum", "mean", "max"])

    t_pulse_features_gr.columns = ["dt", "ticker"] + [
        f"{col[0]}_{col[1]}" for col in t_pulse_features_gr.columns[2:]
    ]

    t_pulse_features_gr = t_pulse_features_gr.reset_index(drop=True)
    t_pulse_features_gr = t_pulse_features_gr.drop(
        columns=[
            col
            for col in t_pulse_features_gr.columns
            if ("top_words_pct" in col and "sum" in col)
        ]
    )
    t_pulse_features_gr["dt"] = pd.to_datetime(t_pulse_features_gr["dt"])

    return t_pulse_features_gr


# Функция feature engineering с макрофакторами (в т.ч. дивидендами и индексами)
def data_from_macrofactors(
    ticker: str, left_date: str, right_date: str, conn
) -> pd.DataFrame:
    dt_left = datetime.strptime(left_date, "%Y-%m-%d").date()
    dt_right = datetime.strptime(right_date, "%Y-%m-%d").date()

    left_date_updated = (dt_left - timedelta(days=shift_days + 1)).strftime("%Y-%m-%d")
    right_date_str = dt_right.strftime("%Y-%m-%d")

    days_list = list(range(1, shift_days + 2))

    indices_query = """
        SELECT * FROM moex_iss_indices
        WHERE date >= %s AND date <= %s
        ORDER BY date ASC
    """
    indices_data = pd.read_sql(
        indices_query, conn, params=(left_date_updated, right_date_str)
    )

    if not indices_data.empty:
        indices_data = indices_data[indices_data["index_code"].isin(indices_list)]
        if not indices_data.empty:
            df_pivot_open = indices_data.pivot(
                index="date", columns="index_code", values="open"
            )
            df_pivot_close = indices_data.pivot(
                index="date", columns="index_code", values="close"
            )

            indices_data_pivot = (
                pd.concat(
                    [
                        df_pivot_open.add_suffix("_open"),
                        df_pivot_close.add_suffix("_close"),
                    ],
                    axis=1,
                )
                .reset_index()
                .rename(columns={"date": "dt"})
            )
        else:
            indices_data_pivot = pd.DataFrame(columns=["dt"])
    else:
        indices_data_pivot = pd.DataFrame(columns=["dt"])

    div_query = """
        SELECT date, ticker, dividedends_rub_per_share as dividends_rub_per_share
        FROM moex_iss_dividends
        WHERE ticker = %s
    """
    dividends_data = pd.read_sql(div_query, conn, params=(ticker,))

    if not dividends_data.empty:
        dividends_data["date"] = pd.to_datetime(dividends_data["date"])
        dividends_data = dividends_data.sort_values("date")

        date_range = pd.date_range(start=left_date_updated, end=right_date_str)
        dividends_data_full = pd.DataFrame({"dt": date_range})
        dividends_data_full["ticker"] = ticker

        dividends_data_full = dividends_data_full.merge(
            dividends_data, how="left", left_on="dt", right_on="date"
        )
        dividends_data_full["dividends_rub_per_share"] = dividends_data_full[
            "dividends_rub_per_share"
        ].ffill()
        dividends_data_full = dividends_data_full.drop(
            columns=["date"], errors="ignore"
        )
    else:
        date_range = pd.date_range(start=left_date_updated, end=right_date_str)
        dividends_data_full = pd.DataFrame(
            {"dt": date_range, "ticker": ticker, "dividends_rub_per_share": np.nan}
        )

    cbrf_query = """
        SELECT * FROM cbrf_data
        WHERE date >= %s AND date <= %s
    """
    min_date_cbrf = datetime.strptime(left_date_updated, "%Y-%m-%d").date() - timedelta(
        days=30
    )
    cbrf_data = pd.read_sql(
        cbrf_query, conn, params=(min_date_cbrf.strftime("%Y-%m-%d"), right_date_str)
    )

    if not cbrf_data.empty:
        cbrf_data = cbrf_data.rename(columns={"date": "dt"})
        date_range_cbrf = pd.date_range(start=min_date_cbrf, end=dt_right)
        cbrf_data_full = pd.DataFrame({"dt": date_range_cbrf}).merge(
            cbrf_data, how="left", on="dt"
        )
        cbrf_data_full = cbrf_data_full.sort_values("dt").ffill()
        cbrf_data_full = cbrf_data_full[
            (cbrf_data_full["dt"] >= left_date_updated)
            & (cbrf_data_full["dt"] <= right_date_str)
        ]
    else:
        date_range_cbrf = pd.date_range(start=left_date_updated, end=right_date_str)
        cbrf_data_full = pd.DataFrame({"dt": date_range_cbrf})

    if not indices_data_pivot.empty:
        indices_data_pivot["dt"] = pd.to_datetime(indices_data_pivot["dt"])
    dividends_data_full["dt"] = pd.to_datetime(dividends_data_full["dt"])
    cbrf_data_full["dt"] = pd.to_datetime(cbrf_data_full["dt"])

    result_df = indices_data_pivot.merge(
        dividends_data_full, how="left", on="dt"
    ).merge(cbrf_data_full, how="left", on="dt")

    result_df["ticker"] = ticker

    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    calc_cols = [col for col in numeric_cols if col not in ["dt"]]

    for col in calc_cols:
        for num_days in days_list:
            result_df[f"{col}_lag_{num_days}"] = result_df[col].shift(num_days)
            result_df[f"{col}_rm_{num_days}"] = result_df[col].rolling(num_days).mean()

    col_to_drop = [col for col in result_df.columns if col.endswith("rm_1")]
    if col_to_drop:
        result_df = result_df.drop(columns=col_to_drop)

    result_df = result_df[
        (result_df["dt"] >= left_date) & (result_df["dt"] <= right_date)
    ].reset_index(drop=True)

    return result_df


# Функция feature engineering со свечами
def data_from_ticker(
    ticker: str, left_date: str, right_date: str, conn
) -> pd.DataFrame:
    dt_left = datetime.strptime(left_date, "%Y-%m-%d").date()
    dt_right = datetime.strptime(right_date, "%Y-%m-%d").date()
    left_date_updated = (dt_left - timedelta(days=shift_days + 1)).strftime("%Y-%m-%d")
    right_date_str = dt_right.strftime("%Y-%m-%d")

    days_list = list(range(1, shift_days + 2))
    max_shift = max(days_list)

    query = """
        SELECT * FROM candles
        WHERE ticker = %s
        AND datetime >= %s AND datetime <= %s
        ORDER BY datetime ASC
    """
    shares = pd.read_sql(
        query, conn, params=(ticker, left_date_updated, right_date_str)
    )

    if shares.empty:
        return pd.DataFrame()

    shares = shares.drop_duplicates()
    shares_features = shares[
        ["datetime", "ticker", "open", "close", "high", "low", "volume"]
    ].rename(columns={"datetime": "dt"})

    numeric_cols = [
        col for col in shares_features.columns if col not in ["ticker", "dt", "close"]
    ]

    for col in numeric_cols:
        for num_days in days_list:
            shares_features[f"{col}_lag_{num_days}"] = shares_features[col].shift(
                num_days
            )
            shares_features[f"{col}_rm_{num_days}"] = (
                shares_features[col].rolling(num_days).mean()
            )

    shares_features[f"close_lag_{max_shift}"] = shares_features["close"].shift(
        max_shift
    )

    for num_days in days_list:
        h_lag = shares_features[f"high_lag_{num_days}"]
        l_lag = shares_features[f"low_lag_{num_days}"]

        diff = h_lag - l_lag
        shares_features[f"high_lag_{num_days}_low_lag_{num_days}_diff"] = diff
        shares_features[f"high_lag_{num_days}_low_lag_{num_days}_diff_pct"] = np.round(
            100 * diff / l_lag.replace(0, np.nan), 3
        )

    h_curr = shares_features["high"]
    l_curr = shares_features["low"]
    diff_curr = h_curr - l_curr
    shares_features["high_low_diff"] = diff_curr
    shares_features["high_low_diff_pct"] = np.round(
        100 * diff_curr / l_curr.replace(0, np.nan), 3
    )

    o_lag = shares_features[f"open_lag_{max_shift}"]
    c_lag = shares_features[f"close_lag_{max_shift}"]
    diff_oc = c_lag - o_lag
    shares_features[f"open_lag_{max_shift}_close_lag_{max_shift}_diff"] = diff_oc
    shares_features[f"open_lag_{max_shift}_close_lag_{max_shift}_diff_pct"] = np.round(
        100 * diff_oc / o_lag.replace(0, np.nan), 3
    )

    stock = StockDataFrame.retype(
        shares_features[["open", "high", "low", "close", "volume"]]
    )

    shares_features[f"macd_lag_{max_shift}"] = np.round(
        stock["macd"].shift(max_shift), 3
    )
    shares_features[f"rsi_14_lag_{max_shift}"] = np.round(
        stock["rsi_14"].shift(max_shift), 3
    )
    shares_features[f"boll_ub_lag_{max_shift}"] = np.round(
        stock["boll_ub"].shift(max_shift), 3
    )
    shares_features[f"boll_lb_lag_{max_shift}"] = np.round(
        stock["boll_lb"].shift(max_shift), 3
    )

    shares_features["target"] = shares_features["close"]
    col_to_drop = [col for col in shares_features.columns if col.endswith("rm_1")]
    shares_features = shares_features.drop(columns=["close"] + col_to_drop)

    shares_features = shares_features.ffill()

    shares_features = shares_features[
        (shares_features["dt"] >= left_date) & (shares_features["dt"] <= right_date)
    ].reset_index(drop=True)

    return shares_features
