import subprocess
import sys
from dotenv import load_dotenv
import pandas as pd
import os

import time
from datetime import datetime, timedelta

load_dotenv()


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from tinkoff.invest import Client, CandleInterval

    print("tinkoff-investments уже установлен")
except ImportError:
    print("Устанавливаю tinkoff-investments...")
    install_package("tinkoff-investments")
    from tinkoff.invest import Client, CandleInterval

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    print("Устанавливаю psycopg2-binary...")
    install_package("psycopg2-binary")
    import psycopg2
    from psycopg2.extras import execute_batch

ALL_TICKERS = [
    # Банки и финансы
    "SBER",
    "SBERP",
    "VTBR",
    "TCSG",
    "CBOM",
    "BSPB",
    "SFIN",
    # Нефть и газ
    "GAZP",
    "LKOH",
    "ROSN",
    "TATN",
    "TATNP",
    "NVTK",
    "SNGS",
    "SNGSP",
    # Металлы и mining
    "GMKN",
    "PLZL",
    "ALRS",
    "POLY",
    "CHMF",
    "NLMK",
    "MAGN",
    "RUAL",
    # IT и технологии
    "YNDX",
    "OZON",
    "TCSG",
    "TTLK",
    # Ритейл и потребительские товары
    "MGNT",
    "FIXP",
    "X5",
    "LENT",
    "MVID",
    "BELU",
    "AQUA",
    # Телекоммуникации
    "MTSS",
    "RTKM",
    "RTKMP",
    # Энергетика и коммунальные услуги
    "IRAO",
    "HYDR",
    "FEES",
    "UPRO",
    "OGKB",
    "MRKC",
    "MRKP",
    # Транспорт и логистика
    "AFLT",
    "FLOT",
    "NMTP",
    # Химия и удобрения
    "PHOR",
    "AKRN",
    # Разное
    "PIKK",
    "MOEX",
    "AFKS",
    "LSRG",
    "RASP",
    "SVAV",
    "ENPG",
    "SMLT",
    "POSI",
    "MDMG",
    "ASTR",
    "UWGN",
    "TRMK",
    "RENI",
    "SOFL",
    "SGZH",
    "SELG",
    "LEAS",
    "VSMO",
    "IRKT",
]

# Словарь с названиями компаний
TICKER_TO_COMPANY = {
    "SBER": "ПАО Сбербанк",
    "SBERP": "ПАО Сбербанк (п)",
    "GAZP": "ПАО Газпром",
    "LKOH": "ПАО Лукойл",
    "GMKN": "ПАО ГМК Норникель",
    "ROSN": "ПАО Роснефть",
    "YNDX": "ПАО Яндекс",
    "TATN": "ПАО Татнефть",
    "TATNP": "ПАО Татнефть (п)",
    "VTBR": "ПАО Банк ВТБ",
    "NVTK": "ПАО Новатэк",
    "ALRS": "ПАО АЛРОСА",
    "POLY": "ПАО Polymetal",
    "PLZL": "ПАО Полюс",
    "MGNT": "ПАО Магнит",
    "MTSS": "ПАО МТС",
    "RTKM": "ПАО Ростелеком",
    "RTKMP": "ПАО Ростелеком (п)",
    "HYDR": "ПАО РусГидро",
    "FEES": "ПАО ФСК ЕЭС",
    "AFKS": "ПАО АФК Система",
    "MOEX": "ПАО Московская биржа",
    "SNGS": "ПАО Сургутнефтегаз",
    "SNGSP": "ПАО Сургутнефтегаз (п)",
    "BSPB": "ПАО Банк Санкт-Петербург",
    "CBOM": "ПАО МКБ",
    "TCSG": "ПАО TCS Group",
    "OZON": "ПАО Ozon",
    "PIKK": "ПАО ПИК",
    "LSRG": "ПАО ЛСР",
    "CHMF": "ПАО Северсталь",
    "NLMK": "ПАО НЛМК",
    "MAGN": "ПАО ММК",
    "RUAL": "ПАО Русал",
    "PHOR": "ПАО ФосАгро",
    "AKRN": "ПАО Акрон",
    "AFLT": "ПАО Аэрофлот",
    "FLOT": "ПАО Совкомфлот",
    "NMTP": "ПАО НМТП",
    "IRAO": "ПАО Интер РАО",
    "UPRO": "ПАО Юнипро",
    "OGKB": "ПАО ОГК-2",
    "MRKC": "ПАО Россети Центр",
    "MRKP": "ПАО Россети Центр и Приволжье",
    "FIXP": "ПАО Fix Price",
    "X5": "ПАО X5 Retail Group",
    "LENT": "ПАО Лента",
    "MVID": "ПАО М.Видео",
    "BELU": "ПАО НоваБев Групп",
    "AQUA": "ПАО ИнАрктика",
    "SMLT": "ГК Самолет",
    "POSI": "Группа Позитив",
    "MDMG": "МКПАО МД Медикал Груп",
    "ASTR": "Группа Астра",
    "UWGN": "ПАО НПК ОВК",
    "TRMK": "ПАО ТМК",
    "RENI": "ПАО Ренессанс Страхование",
    "SOFL": "ПАО Софтлайн",
    "SGZH": "ПАО Сегежа",
    "SELG": "ПАО Селигдар",
    "LEAS": "ПАО ЛК Европлан",
    "VSMO": "ПАО ВСМПО-АВИСМА",
    "IRKT": "ПАО Иркут",
    "RASP": "ПАО Распадская",
    "SVAV": "ПАО Соллерс",
    "ENPG": "МКПАО Эн+ Груп",
    "TTLK": "ПАО Т-Технологии",
}


class DatabaseManager:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Подключение к базе данных"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ Успешное подключение к PostgreSQL")
        except Exception as e:
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise

    def create_tables(self):
        """Создание таблиц если они не существуют"""
        try:
            with self.conn.cursor() as cursor:
                # Таблица компаний
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(20) UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        figi VARCHAR(50) UNIQUE NOT NULL,
                        currency VARCHAR(10),
                        lot INTEGER,
                        min_price_increment DECIMAL(10,6),
                        sector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Таблица свечей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(20) REFERENCES companies(ticker),
                        datetime TIMESTAMP NOT NULL,
                        open DECIMAL(15,6),
                        high DECIMAL(15,6),
                        low DECIMAL(15,6),
                        close DECIMAL(15,6),
                        volume BIGINT,
                        is_complete BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, datetime)
                    )
                """)

                # Таблица метаданных сборов
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS collection_metadata (
                        id SERIAL PRIMARY KEY,
                        collection_timestamp TIMESTAMP NOT NULL,
                        total_searched INTEGER NOT NULL,
                        found_stocks INTEGER NOT NULL,
                        not_found_count INTEGER NOT NULL,
                        collection_years INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Таблица не найденных тикеров
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS not_found_tickers (
                        id SERIAL PRIMARY KEY,
                        collection_id INTEGER REFERENCES collection_metadata(id),
                        ticker VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Индексы для оптимизации
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_candles_ticker_datetime ON candles(ticker, datetime)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_candles_datetime ON candles(datetime)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker)"
                )

                self.conn.commit()
                print("✅ Таблицы созданы/проверены")

        except Exception as e:
            print(f"❌ Ошибка создания таблиц: {e}")
            self.conn.rollback()
            raise

    def save_company_info(self, company_info):
        """Сохранение информации о компании"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO companies (ticker, name, figi, currency, lot, min_price_increment, sector)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        name = EXCLUDED.name,
                        figi = EXCLUDED.figi,
                        currency = EXCLUDED.currency,
                        lot = EXCLUDED.lot,
                        min_price_increment = EXCLUDED.min_price_increment,
                        sector = EXCLUDED.sector,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (
                        company_info["ticker"],
                        company_info["name"],
                        company_info["figi"],
                        company_info["currency"],
                        company_info["lot"],
                        company_info["min_price_increment"],
                        company_info["sector"],
                    ),
                )
                self.conn.commit()
                return True
        except Exception as e:
            print(f"❌ Ошибка сохранения компании {company_info['ticker']}: {e}")
            self.conn.rollback()
            return False

    def save_candles_batch(self, ticker, candles_df):
        """Пакетное сохранение свечей"""
        if candles_df.empty:
            return True

        try:
            with self.conn.cursor() as cursor:
                # Подготавливаем данные для вставки
                data_tuples = [
                    (
                        ticker,
                        row["datetime"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["is_complete"],
                    )
                    for _, row in candles_df.iterrows()
                ]

                # Пакетная вставка
                execute_batch(
                    cursor,
                    """
                    INSERT INTO candles (ticker, datetime, open, high, low, close, volume, is_complete)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, datetime) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        is_complete = EXCLUDED.is_complete
                """,
                    data_tuples,
                )

                self.conn.commit()
                print(f"   📊 Сохранено {len(data_tuples)} свечей для {ticker}")
                return True

        except Exception as e:
            print(f"❌ Ошибка сохранения свечей для {ticker}: {e}")
            self.conn.rollback()
            return False

    def save_collection_metadata(self, metadata, not_found_tickers):
        """Сохранение метаданных сбора"""
        try:
            with self.conn.cursor() as cursor:
                # Сохраняем основную метаинформацию
                cursor.execute(
                    """
                    INSERT INTO collection_metadata 
                    (collection_timestamp, total_searched, found_stocks, not_found_count, collection_years)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        metadata["timestamp"],
                        metadata["total_searched"],
                        metadata["found_stocks"],
                        metadata["not_found_count"],
                        metadata["collection_years"],
                    ),
                )

                collection_id = cursor.fetchone()[0]

                # Сохраняем не найденные тикеры
                if not_found_tickers:
                    not_found_tuples = [
                        (collection_id, ticker) for ticker in not_found_tickers
                    ]
                    execute_batch(
                        cursor,
                        """
                        INSERT INTO not_found_tickers (collection_id, ticker)
                        VALUES (%s, %s)
                    """,
                        not_found_tuples,
                    )

                self.conn.commit()
                print(f"✅ Метаданные сохранены (ID: {collection_id})")
                return collection_id

        except Exception as e:
            print(f"❌ Ошибка сохранения метаданных: {e}")
            self.conn.rollback()
            return None

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            print("✅ Соединение с PostgreSQL закрыто")


class CompleteDataCollector:
    def __init__(self, token):
        self.token = token

    def find_available_stocks(self):
        """Поиск доступных акций из списка"""
        print(f"Всего тикеров в списке: {len(ALL_TICKERS)}")

        with Client(self.token) as client:
            try:
                all_shares = client.instruments.shares().instruments
                print(f"Всего акций в Tinkoff API: {len(all_shares)}")

                available_stocks = []
                not_found = []

                # Создаем словарь для быстрого поиска
                shares_by_ticker = {}
                for share in all_shares:
                    shares_by_ticker[share.ticker] = share

                for ticker in ALL_TICKERS:
                    if ticker in shares_by_ticker:
                        share = shares_by_ticker[ticker]

                        if share.currency == "rub" and share.buy_available_flag:
                            stock_info = {
                                "ticker": share.ticker,
                                "figi": share.figi,
                                "name": TICKER_TO_COMPANY.get(ticker, share.name),
                                "api_name": share.name,
                                "currency": share.currency,
                                "lot": share.lot,
                                "min_price_increment": self._quotation_to_float(
                                    share.min_price_increment
                                ),
                                "sector": share.sector
                                if hasattr(share, "sector")
                                else "Не указан",
                            }
                            available_stocks.append(stock_info)
                            print(f"{ticker}: {stock_info['name']}")
                        else:
                            not_found.append(ticker)
                            print(f"{ticker}: недоступна для торговли")
                    else:
                        not_found.append(ticker)
                        print(f"{ticker}: не найден")

                print("\n ИТОГО:")
                print(f"   Найдено: {len(available_stocks)} акций")
                print(f"   Не найдено: {len(not_found)} акций")

                if not_found:
                    print(f"   Не найденные: {', '.join(not_found)}")

                return available_stocks, not_found

            except Exception as e:
                print(f"Ошибка при поиске акций: {e}")
                return [], ALL_TICKERS

    def collect_extended_data(self, stocks_info, years=10):
        """Сбор данных с расширенной историей"""
        all_data = {}
        failed_tickers = []

        print(f"\n Начинаем сбор данных за {years} лет...")
        print(f" Обрабатываем {len(stocks_info)} акций")
        print("-" * 60)

        for i, stock in enumerate(stocks_info, 1):
            try:
                print(f" [{i:2d}/{len(stocks_info)}] {stock['ticker']}...")

                # Собираем данные по частям
                candles_df = self._collect_data_in_chunks(stock["figi"], years=years)

                if candles_df is not None and len(candles_df) > 100:
                    all_data[stock["ticker"]] = {
                        "data": candles_df,
                        "info": stock,
                        "first_date": candles_df["datetime"].min(),
                        "last_date": candles_df["datetime"].max(),
                        "candle_count": len(candles_df),
                        "data_quality": self._assess_data_quality(candles_df),
                        "period_years": (
                            candles_df["datetime"].max() - candles_df["datetime"].min()
                        ).days
                        / 365.25,
                    }

                    print(f"   Успешно: {len(candles_df)} свечей")
                    print(
                        f"   Период: {candles_df['datetime'].min().date()} - {candles_df['datetime'].max().date()}"
                    )

                else:
                    failed_tickers.append(stock["ticker"])
                    print(
                        f"   Мало данных: {len(candles_df) if candles_df else 0} свечей"
                    )

                # Пауза между запросами
                time.sleep(0.3)

            except Exception as e:
                failed_tickers.append(stock["ticker"])
                print(f"   Ошибка: {e}")
                continue

        print("-" * 60)
        print(f"Успешно собрано: {len(all_data)} акций")
        if failed_tickers:
            print(f" Проблемы с: {', '.join(failed_tickers)}")

        return all_data

    def _collect_data_in_chunks(self, figi, years=10, chunk_years=3):
        """Сбор данных частями"""
        all_chunks = []
        end_time = datetime.utcnow()

        # Разбиваем на периоды
        for chunk_start in range(0, years, chunk_years):
            chunk_end = min(chunk_start + chunk_years, years)

            start_time = end_time - timedelta(days=chunk_end * 365)
            chunk_start_time = end_time - timedelta(days=chunk_start * 365)

            try:
                chunk_data = self._get_candles_period(
                    figi, start_time, chunk_start_time
                )
                if chunk_data is not None and len(chunk_data) > 0:
                    all_chunks.append(chunk_data)

                time.sleep(0.2)

            except Exception:
                continue

        # Объединяем все чанки
        if all_chunks:
            combined_df = pd.concat(all_chunks, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["datetime"]).sort_values(
                "datetime"
            )
            return combined_df
        else:
            return pd.DataFrame()

    def _get_candles_period(
        self, figi, start_time, end_time, interval=CandleInterval.CANDLE_INTERVAL_DAY
    ):
        """Получение свечей за период"""
        with Client(self.token) as client:
            try:
                candles = client.get_all_candles(
                    figi=figi, from_=start_time, to=end_time, interval=interval
                )
                return self._candles_to_dataframe(candles)
            except Exception:
                return pd.DataFrame()

    def _candles_to_dataframe(self, candles):
        """Конвертация свечей в DataFrame"""
        data = []
        for candle in candles:
            data.append(
                {
                    "datetime": candle.time,
                    "open": self._quotation_to_float(candle.open),
                    "high": self._quotation_to_float(candle.high),
                    "low": self._quotation_to_float(candle.low),
                    "close": self._quotation_to_float(candle.close),
                    "volume": candle.volume,
                    "is_complete": candle.is_complete,
                }
            )
        return pd.DataFrame(data) if data else pd.DataFrame()

    def _quotation_to_float(self, quotation):
        """Конвертация Quotation в float"""
        if hasattr(quotation, "units") and hasattr(quotation, "nano"):
            return quotation.units + quotation.nano / 1e9
        return float(quotation)

    def _assess_data_quality(self, df):
        """Оценка качества данных"""
        if len(df) == 0:
            return "Нет данных"

        missing_data = df[["open", "high", "low", "close"]].isnull().sum().sum()
        zero_volume = (df["volume"] == 0).sum()

        quality_score = (len(df) - missing_data - zero_volume / 10) / len(df)

        if quality_score > 0.95:
            return "Отличное"
        elif quality_score > 0.8:
            return "Хорошее"
        elif quality_score > 0.6:
            return "Удовлетворительное"
        else:
            return "Плохое"


class CompleteDataDBManager:
    def __init__(self, db_config):
        self.db_manager = DatabaseManager(db_config)

    def save_all_data(
        self, all_data, available_stocks, not_found_tickers, collection_years
    ):
        """Сохранение всех данных в базу данных"""
        timestamp = datetime.now()

        print("\n Сохранение данных в PostgreSQL...")

        # Сохранение информации о компаниях
        companies_saved = 0
        for stock in available_stocks:
            if self.db_manager.save_company_info(stock):
                companies_saved += 1
        print(f"   Компании сохранены: {companies_saved}/{len(available_stocks)}")

        # Сохранение свечей
        candles_saved = 0
        for ticker, data in all_data.items():
            if self.db_manager.save_candles_batch(ticker, data["data"]):
                candles_saved += 1
        print(f"   Свечи сохранены: {candles_saved}/{len(all_data)}")

        # Сохранение метаданных
        metadata = {
            "timestamp": timestamp,
            "total_searched": len(ALL_TICKERS),
            "found_stocks": len(all_data),
            "not_found_count": len(not_found_tickers),
            "collection_years": collection_years,
        }

        collection_id = self.db_manager.save_collection_metadata(
            metadata, not_found_tickers
        )

        # Создание аналитического отчета
        self._create_analysis_report(all_data, timestamp)

        return timestamp, collection_id

    def _create_analysis_report(self, all_data, timestamp):
        """Создание аналитического отчета"""
        analysis_data = []

        for ticker, data in all_data.items():
            df = data["data"]
            if len(df) > 0:
                returns = df["close"].pct_change()

                analysis_data.append(
                    {
                        "ticker": ticker,
                        "name": data["info"]["name"],
                        "sector": data["info"]["sector"],
                        "first_date": data["first_date"].strftime("%Y-%m-%d"),
                        "last_date": data["last_date"].strftime("%Y-%m-%d"),
                        "total_candles": len(df),
                        "period_years": round(data["period_years"], 1),
                        "start_price": df["close"].iloc[0],
                        "end_price": df["close"].iloc[-1],
                        "total_return_percent": (
                            df["close"].iloc[-1] - df["close"].iloc[0]
                        )
                        / df["close"].iloc[0]
                        * 100,
                        "max_price": df["high"].max(),
                        "min_price": df["low"].min(),
                        "avg_daily_volume": df["volume"].mean(),
                        "volatility_percent": returns.std() * 100
                        if len(returns) > 1
                        else 0,
                        "data_quality": data["data_quality"],
                    }
                )

        analysis_df = pd.DataFrame(analysis_data)

        # Вывод сводки в консоль
        print("\n СВОДКА ПО ВСЕМ КОМПАНИЯМ:")
        print("=" * 60)
        print(f" Акций собрано: {len(analysis_df)}")
        print(f" Всего свечей: {analysis_df['total_candles'].sum():,}")
        print(f" Средний период: {analysis_df['period_years'].mean():.1f} лет")

        # Топ-5 по доходности
        if len(analysis_df) > 0:
            print("\n ТОП-5 ПО ДОХОДНОСТИ:")
            top_return = analysis_df.nlargest(5, "total_return_percent")[
                ["ticker", "name", "total_return_percent"]
            ]
            for _, row in top_return.iterrows():
                symbol = "🟢" if row["total_return_percent"] > 0 else "🔴"
                print(
                    f"   {symbol} {row['ticker']}: {row['total_return_percent']:+.1f}%"
                )

    def close_connection(self):
        """Закрытие соединения с базой данных"""
        self.db_manager.close()


def show_database_stats(db_config):
    """Показать статистику базы данных"""
    try:
        db_manager = DatabaseManager(db_config)

        with db_manager.conn.cursor() as cursor:
            # Статистика компаний
            cursor.execute("SELECT COUNT(*) FROM companies")
            companies_count = cursor.fetchone()[0]

            # Статистика свечей
            cursor.execute("SELECT COUNT(*) FROM candles")
            candles_count = cursor.fetchone()[0]

            # Статистика по датам
            cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM candles")
            min_date, max_date = cursor.fetchone()

            # Статистика по сборкам
            cursor.execute(
                "SELECT COUNT(*), MAX(collection_timestamp) FROM collection_metadata"
            )
            collections_count, last_collection = cursor.fetchone()

        db_manager.close()

        print("\n СТАТИСТИКА БАЗЫ ДАННЫХ:")
        print("-" * 50)
        print(f" Компаний: {companies_count}")
        print(f" Свечей: {candles_count:,}")
        print(f" Период данных: {min_date.date()} - {max_date.date()}")
        print(f" Сборок данных: {collections_count}")
        print(f" Последняя сборка: {last_collection}")

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")


def main_complete_collection():
    # Конфигурация базы данных PostgreSQL
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }

    # ТОКЕН Tinkoff
    TINKOFF_TOKEN = "TOKEN"

    # Настройки
    COLLECTION_YEARS = 10  # Лет истории

    print(" ПОЛНЫЙ СБОР ДАННЫХ В POSTGRESQL")
    print("=" * 60)
    print(f" Компаний в списке: {len(ALL_TICKERS)}")
    print(f" Период сбора: {COLLECTION_YEARS} лет")
    print("=" * 60)

    try:
        # Инициализация
        collector = CompleteDataCollector(TINKOFF_TOKEN)
        data_manager = CompleteDataDBManager(DB_CONFIG)

        # Шаг 1: Поиск доступных акций
        available_stocks, not_found = collector.find_available_stocks()

        if not available_stocks:
            print("❌ Не найдено доступных акций!")
            return

        # Шаг 2: Сбор данных
        print(f"\n Начинаем сбор данных за {COLLECTION_YEARS} лет...")
        all_data = collector.collect_extended_data(
            available_stocks, years=COLLECTION_YEARS
        )

        if all_data:
            # Шаг 3: Сохранение в базу данных
            timestamp, collection_id = data_manager.save_all_data(
                all_data, available_stocks, not_found, COLLECTION_YEARS
            )

            print("\n ПОЛНЫЙ СБОР ДАННЫХ ЗАВЕРШЕН!")
            print(" Данные сохранены в PostgreSQL")
            print(f" ID сборки: {collection_id}")

            # Показываем статистику
            show_database_stats(DB_CONFIG)

        else:
            print(" Не удалось собрать данные")

        # Закрытие соединения
        data_manager.close_connection()

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


def quick_db_stats(db_config):
    """Быстрая статистика базы данных"""
    try:
        db_manager = DatabaseManager(db_config)

        with db_manager.conn.cursor() as cursor:
            # Основная статистика
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT ticker) as companies_count,
                    COUNT(*) as candles_count,
                    MIN(datetime) as first_date,
                    MAX(datetime) as last_date
                FROM candles
            """)
            stats = cursor.fetchone()

            # Распределение по компаниям
            cursor.execute("""
                SELECT ticker, COUNT(*) as candle_count
                FROM candles 
                GROUP BY ticker 
                ORDER BY candle_count DESC
                LIMIT 10
            """)
            top_companies = cursor.fetchall()

        db_manager.close()

        print("\n БЫСТРАЯ СТАТИСТИКА:")
        print(f" Компаний: {stats[0]}")
        print(f" Свечей: {stats[1]:,}")
        print(f" Период: {stats[2].date()} - {stats[3].date()}")

        print("\n ТОП-10 по количеству свечей:")
        for ticker, count in top_companies:
            print(f"   {ticker}: {count:,} свечей")

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")


if __name__ == "__main__":
    # Конфигурация базы данных
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }

    # Основной сбор данных
    main_complete_collection()

    # Дополнительно: быстрая статистика
    quick_db_stats(DB_CONFIG)
