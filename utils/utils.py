# Импорты dev-ops
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Импорты data-analysis
import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from datetime import datetime, timedelta

# Импорты nlp
import re
import ast
import spacy
from razdel import sentenize
from collections import Counter
from nltk.corpus import stopwords

# Импорты others
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Импорт variables
from .config_ml import TRAIN_PERIOD, VAL_PERIOD, TEST_PERIOD, TOP_100_WORDS, INDICES
load_dotenv()

# Глобальные переменные
shift_days = max(TRAIN_PERIOD, VAL_PERIOD, TEST_PERIOD)
top_words_list = TOP_100_WORDS
indices_list = INDICES
nlp = spacy.load("ru_core_news_sm")
russian_stopwords = set(list(set(stopwords.words("russian"))) + ['это', 'р', 'г', 'руб', 'шт'])

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
    text = re.sub(r"\d+", " ", text)      # убираем цифры
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_reactions(reactions_str):
    try:
        reactions = ast.literal_eval(reactions_str)
    except:
        reactions = []
    reaction_dict = {f"reaction_{r['type']}": r['count'] for r in reactions if 'type' in r and 'count' in r}
    for r in ['buy-up','rocket','not-convinced','get-rid','like','dislike']:
        reaction_dict.setdefault(f"reaction_{r}", 0)
    reaction_dict['total_reactions'] = sum(reaction_dict.values())
    return pd.Series(reaction_dict)

def lemmatize_no_stop(text):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.text not in russian_stopwords and t.is_alpha]

def lemmatize_no_stop_batch(texts, batch_size):
    """Лемматизация батчами для экономии памяти"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        docs = list(nlp.pipe(batch))
        lemmas = [[t.lemma_ for t in doc if t.text not in russian_stopwords and t.is_alpha]
                  for doc in docs]
        results.extend(lemmas)
    return results

def top_words_stats(lemmas):
    total_words = len(lemmas)
    counts = Counter(lemmas)
    stats = {}
    for word in top_words_list:
        stats[f"tfidf_{word}"] = counts.get(word, 0) / total_words if total_words > 0 else 0
    stats['top_words_pct'] = sum([stats[f"tfidf_{w}"] for w in top_words_list])
    return pd.Series(stats)

# Функция feature engineering с Т-Пульс
def data_from_tpulse(ticker: str, left_date: str, right_date: str, conn) -> pd.DataFrame:
    t_pulse_data = pd.read_sql(
        f"""
        SELECT * FROM t_pulse_data
        WHERE ticker = '{ticker}'
        AND inserted >= '{left_date}' AND inserted <= '{right_date}'
        ORDER BY inserted ASC
    """, conn)
    t_pulse_features = t_pulse_data.drop(columns=['id']).copy()
    t_pulse_features = t_pulse_features.rename(columns={'inserted':'dt'})
    t_pulse_features['dt'] = pd.to_datetime(t_pulse_features['dt'], errors='coerce').dt.date

    # Обработка текста
    t_pulse_features['clean_text'] = t_pulse_features['text'].apply(clean_text)
    t_pulse_features['lemmas'] = lemmatize_no_stop_batch(t_pulse_features['clean_text'].tolist(), batch_size=100)
    #t_pulse_features['clean_text'].apply(lemmatize_no_stop)

    # Формирование базовых текстовых фичей
    t_pulse_features['num_words'] = t_pulse_features['lemmas'].apply(len)
    t_pulse_features['num_chars'] = t_pulse_features['clean_text'].apply(len)

    # Обработка реакций
    reactions_df = t_pulse_features['reactions_counters'].apply(parse_reactions)
    t_pulse_features = pd.concat([t_pulse_features, reactions_df], axis=1)

    # Обработка числа комментариев
    t_pulse_features['commentscount'] = pd.to_numeric(t_pulse_features['commentscount'], errors='coerce').fillna(0)
    t_pulse_features['comment_strength'] = np.log(np.exp(1) + t_pulse_features['commentscount'])

    # Формирование фичй tf-idf по самым важным словам в постах
    top_words_df = t_pulse_features['lemmas'].apply(top_words_stats)
    t_pulse_features = pd.concat([t_pulse_features, top_words_df], axis=1)
    t_pulse_features = t_pulse_features.drop(columns=['text', 'clean_text', 'lemmas', 'reactions_counters'])

    # Агрегация по ранее сформированным фичам
    agg_cols = t_pulse_features.drop(columns=['dt', 'ticker']).columns
    t_pulse_features_gr = t_pulse_features.groupby(['dt','ticker'], as_index=False)[agg_cols].agg(['sum','mean','max'])
    t_pulse_features_gr.columns = ['dt', 'ticker'] + [f"{col[0]}_{col[1]}" for col in t_pulse_features_gr.columns[2:]]

    t_pulse_features_gr = t_pulse_features_gr.reset_index(drop=True)
    t_pulse_features_gr = t_pulse_features_gr.drop(columns=[col for col in t_pulse_features_gr.columns if ('top_words_pct' in col and 'sum' in col)])
    t_pulse_features_gr['dt'] = pd.to_datetime(t_pulse_features_gr['dt'])

    return t_pulse_features_gr

# Функция feature engineering с макрофакторами (в т.ч. дивидендами и индексами)
def data_from_macrofactors(ticker: str, left_date: str, right_date: str, conn) -> pd.DataFrame:
    left_date_updated = (datetime.strptime(left_date, '%Y-%m-%d').date() - timedelta(days=shift_days+1)).strftime('%Y-%m-%d')
    days_list = [num_days for num_days in range(1, shift_days + 2)]

    indices_data = pd.read_sql(
        f"""
        SELECT * FROM moex_iss_indices
        WHERE date >= '{left_date_updated}' AND date <= '{right_date}'
        ORDER BY date ASC
        """, conn)

    # Создание pivot таблиц: index_code / open -> index_code_open
    indices_data = indices_data[indices_data['index_code'].isin(indices_list)]
    df_pivot_open = indices_data.pivot(index='date', columns='index_code', values='open')
    df_pivot_close = indices_data.pivot(index='date', columns='index_code', values='close')
    indices_data_pivot = pd.concat([df_pivot_open.add_suffix('_open'), df_pivot_close.add_suffix('_close')], axis=1)
    indices_data_pivot = indices_data_pivot.reset_index()
    indices_data_pivot = indices_data_pivot.rename(columns={'date':'dt'})

    dividends_data = pd.read_sql(
        f"""
        SELECT date, ticker, dividedends_rub_per_share as dividends_rub_per_share
        FROM moex_iss_dividends
        WHERE ticker = '{ticker}'
        """, conn)

    dividends_data['date'] = pd.to_datetime(dividends_data['date'])

    # Формирование датафрейма с заполненными по дате дивидендами
    date_range = pd.date_range(start=datetime.strptime(left_date_updated, '%Y-%m-%d').date(),
                               end=datetime.strptime(right_date, '%Y-%m-%d').date())

    dividends_data_full = pd.DataFrame({'dt': date_range})
    dividends_data_full['ticker'] = ticker
    dividends_data = dividends_data.sort_values('date')
    dividends_data_full['dividends_rub_per_share'] = dividends_data_full['dt'].apply(lambda x: dividends_data[dividends_data['date'] <= x]['dividends_rub_per_share'].iloc[-1])

    cbrf_data = pd.read_sql(
        f"""
        SELECT *
        FROM cbrf_data
        """, conn)

    # Формирование датафрейма с заполненными по дате макрофакторами из ЦБ РФ
    date_range_cbrf = pd.date_range(start=min(cbrf_data['date']), end=datetime.strptime(right_date, '%Y-%m-%d').date())

    cbrf_data_full = pd.DataFrame({'dt': date_range_cbrf})
    cbrf_data = cbrf_data.rename(columns={'date':'dt'})
    cbrf_data_full = cbrf_data_full.merge(cbrf_data, how='left', on='dt')
    cbrf_data_full = cbrf_data_full.sort_values('dt')
    cbrf_data_full = cbrf_data_full.ffill()
    cbrf_data_full = cbrf_data_full[(cbrf_data_full['dt'] >= left_date_updated) & (cbrf_data_full['dt'] <= right_date)]

    # Соединяем датафреймы-источники: дивиденды, индексы и макрофакторы из ЦБ
    result_df = indices_data_pivot.merge(dividends_data_full, how='left', on='dt').merge(cbrf_data_full, how='left', on='dt')
    for col in result_df.columns:
        if col not in ['ticker', 'dt']:
            for num_days in days_list:
                # Лаги
                result_df[f"{col}_lag_{num_days}"] = result_df[col].shift(num_days)
                # CС
                result_df[f"{col}_rm_{num_days}"] = result_df[col].rolling(num_days).mean()
    col_to_drop = [col for col in result_df.columns if col.endswith('rm_1')]
    result_df = result_df.drop(columns=col_to_drop)
    result_df = result_df[(result_df['dt'] >= left_date) & (result_df['dt'] <= right_date)]

    return result_df

# Функция feature engineering со свечами
def data_from_ticker(ticker: str, left_date: str, right_date: str, conn) -> pd.DataFrame:
    left_date_updated = (datetime.strptime(left_date, '%Y-%m-%d').date() - timedelta(days=shift_days+1)).strftime('%Y-%m-%d')
    days_list = [num_days for num_days in range(1, shift_days + 2)]
    shares = pd.read_sql(
        f"""
        SELECT * FROM candles
        WHERE ticker = '{ticker}'
        AND datetime >= '{left_date_updated}' AND datetime <= '{right_date}'
        ORDER BY datetime ASC
    """, conn)

    shares = shares.drop_duplicates()
    shares_features = shares[['datetime', 'ticker', 'open', 'close', 'high', 'low', 'volume']].copy()
    shares_features = shares_features.rename(columns={'datetime':'dt'})

    # Лаги и скользящее среднее
    for col in shares_features.columns:
        if col not in ['ticker', 'dt', 'close']:
            for num_days in days_list:
                # Лаги
                shares_features[f"{col}_lag_{num_days}"] = shares_features[col].shift(num_days)
                # CС
                shares_features[f"{col}_rm_{num_days}"] = shares_features[col].rolling(num_days).mean()
        elif col == 'close':
            shares_features[f"{col}_lag_{max(days_list)}"] = shares_features[col].shift(max(days_list))

    for num_days in days_list:
        # Доп фичи на основании лагов
        shares_features[f"high_lag_{num_days}_low_lag_{num_days}_diff"] = shares_features[f"high_lag_{num_days}"] - shares_features[f"low_lag_{num_days}"]
        shares_features[f"high_lag_{num_days}_low_lag_{num_days}_diff_pct"] = round(100 * (shares_features[f"high_lag_{num_days}"] - shares_features[f"low_lag_{num_days}"]) / shares_features[f"low_lag_{num_days}"], 3)

    # Аналогичные расчеты для актуальных свечей только БЕЗ close
    shares_features['high_low_diff'] = shares_features['high'] - shares_features['low']
    shares_features['high_low_diff_pct'] = round(100 * (shares_features['high'] - shares_features['low']) / shares_features['low'], 3)

     # На основе фичей с шагом max(days_list) назад создаем свечи:
    shares_features[f"open_lag_{max(days_list)}_close_lag_{max(days_list)}_diff"] = shares_features[f"close_lag_{max(days_list)}"] - shares_features[f"open_lag_{max(days_list)}"]
    shares_features[f"open_lag_{max(days_list)}_close_lag_{max(days_list)}_diff_pct"] = round(100 * (shares_features[f"close_lag_{max(days_list)}"] - shares_features[f"open_lag_{max(days_list)}"]) / shares_features[f"open_lag_{max(days_list)}"], 3)

    # Формирование индикаторов (строятся на сегодня -> отбираем с лагом shift_days назад)
    stock = StockDataFrame.retype(shares_features)
    shares_features[f"macd_lag_{max(days_list)}"] = round(stock["macd"].shift(max(days_list)),3)
    shares_features[f"rsi_14_lag_{max(days_list)}"] = round(stock["rsi_14"].shift(max(days_list)),3)
    shares_features[f"boll_ub_lag_{max(days_list)}"] = round(stock["boll_ub"].shift(max(days_list)),3)
    shares_features[f"boll_lb_lag_{max(days_list)}"] = round(stock["boll_lb"].shift(max(days_list)),3)

    # Формирование таргета
    shares_features["target"] = shares_features["close"]
    col_to_drop = [col for col in shares_features.columns if col.endswith('rm_1')]
    shares_features = shares_features.drop(columns=['close'] + col_to_drop)

    for col in shares_features.columns:
        shares_features[col] = shares_features[col].ffill()

    shares_features = shares_features[(shares_features['dt'] >= left_date) & (shares_features['dt'] <= right_date)]
    shares_features = shares_features.reset_index(drop=True) # .dropna()

    return shares_features