from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Период обучения в днях
TRAIN_PERIOD = 21

# Период валидации в днях (подбор ГП)
VAL_PERIOD = 14

# Период теста
TEST_PERIOD = 7

# Шаг скользящего окна cross_validation (желательно, чтобы был равен или больше test_period)
STEP = 7

# Количество итераций в Optuna
N_TRIALS = 500

# Оптимизируемая метрика в Optuna: MAPE, MAE
METRIC_OPTUNA = "MAPE"

# Будьте внимательны, чем меньше сумма дней train/val/test, тем больше окон, тем меньше фичей войдет в финальный топ по модели
# Нужно балансировать между кол-вом окон и выбранных фичей. Мало окон и много фичей скорее всего не очень хорошо, как и много окон и мало фичей.
# Количество топ-фичей для отбора
TOP_N_FEATURES = 500

# Список ML моделей для обучения и список ГП для оптимизации
MODELS_CONFIG = {
    # Классические деревья
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "optuna_objective": lambda trial: {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        },
    },
    # Случайный лес
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "optuna_objective": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        },
    },
    # Extra Trees (новая модель)
    "ExtraTrees": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "optuna_objective": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        },
    },
    # Градиентный бустинг
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "optuna_objective": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        },
    },
    # HistGradientBoosting (новая быстрая модель)
    "HistGradientBoosting": {
        "model": HistGradientBoostingRegressor(random_state=42, max_iter=100),
        "optuna_objective": lambda trial: {
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        },
    },
    # XGBoost
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "optuna_objective": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        },
    },
    # LightGBM
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "optuna_objective": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        },
    },
    # CatBoost
    "CatBoost": {
        "model": CatBoostRegressor(random_state=42, verbose=False),
        "optuna_objective": lambda trial: {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
        },
    },
    # SVR (новая модель)
    "SVR": {
        "model": SVR(),
        "optuna_objective": lambda trial: {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 0.3),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "gamma": trial.suggest_float("gamma", 0.001, 1.0, log=True),
        },
    },
    # KNeighbors (новая модель)
    "KNeighbors": {
        "model": KNeighborsRegressor(n_jobs=-1),
        "optuna_objective": lambda trial: {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        },
    },
    # Neural Network MLP (новая модель)
    "MLP": {
        "model": MLPRegressor(random_state=42, early_stopping=True, verbose=False),
        "optuna_objective": lambda trial: {
            "hidden_layer_sizes": (
                trial.suggest_int("h1", 50, 200),
                trial.suggest_int("h2", 20, 100),
            ),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "logistic"]
            ),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "adaptive"]
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
        },
    },
    # Stacking Ensemble (новая модель)
    "Stacking": {
        "model": None,  # Создается динамически
        "optuna_objective": None,
    },
}

# Топ 100 слов для TF-IDF по постам Т-Пульс (менять НЕ рекомендуется)
TOP_100_WORDS = [
    "акция",
    "рост",
    "год",
    "цена",
    "рынок",
    "дивиденд",
    "компания",
    "день",
    "уровень",
    "новость",
    "анализ",
    "пока",
    "бумага",
    "рубль",
    "профиль",
    "потенциал",
    "сегодня",
    "пост",
    "неделя",
    "поддержка",
    "дать",
    "пульс",
    "быть",
    "подпишись",
    "портфель",
    "список",
    "млрд",
    "цель",
    "график",
    "текущий",
    "зона",
    "снижение",
    "индекс",
    "месяц",
    "актив",
    "иир",
    "новый",
    "технический",
    "результат",
    "пропустить",
    "хороший",
    "ставка",
    "близкий",
    "следующий",
    "объём",
    "время",
    "банк",
    "итог",
    "идея",
    "оставаться",
    "первый",
    "ждать",
    "прибыль",
    "россия",
    "выше",
    "тренд",
    "движение",
    "общий",
    "индикатор",
    "свой",
    "вывод",
    "позиция",
    "возможный",
    "сопротивление",
    "падение",
    "биржа",
    "такой",
    "последний",
    "отскок",
    "млн",
    "вверх",
    "лонг",
    "сигнал",
    "также",
    "мой",
    "мочь",
    "момент",
    "мсфо",
    "т",
    "весь",
    "стать",
    "покупка",
    "купить",
    "инвестиция",
    "около",
    "высоко",
    "хотеть",
    "очень",
    "вырасти",
    "фактор",
    "финансовый",
    "инвестор",
    "российский",
    "интересный",
    "ещё",
    "вниз",
    "пао",
    "ожидать",
    "которые",
    "отчёт",
] + ["сша", "китай", "ес", "европа"]
# Индексы, которые подгружаются в фичи (выбраны индексы регулярно обновляемые на стороне МосБиржи)
INDICES = [
    "RTSIT",
    "RTSFN",
    "RUMBTRNS",
    "MXSHAR",
    "BPSIG",
    "MOEXMM",
    "RUCBTRNS",
    "RTSCH",
    "IMOEX",
    "MOEXTL",
    "MOEXIT",
    "IMOEXCNY",
    "MRBC",
    "MREFTR",
    "MOEXOG",
    "MOEXCH",
    "RTSTN",
    "RUBMI",
    "EPSITR",
    "MOEXFN",
    "RTSRE",
    "RTSTL",
    "IMOEX2",
    "BPSI",
    "MRSV",
    "RGBITR",
    "MREF",
    "RUPAI",
    "MOEXINN",
    "MRRT",
    "MOEXRE",
    "IMOEXW",
    "RUPMI",
    "RTSI",
    "MCXSM",
    "MOEXCN",
    "RUPCI",
    "RTSCR",
    "MOEXBMI",
    "RUGOLD",
    "MIPO",
    "RTSSM",
    "MOEXEU",
    "MOEX10",
    "MESG",
    "MRSVR",
    "RUABITR",
    "RTSMM",
    "RTSEU",
    "MOEXTN",
    "RTSOG",
    "MOEXBC",
    "RUSFAR1MRT",
    "RUSFAR",
    "RUSFARRT",
    "RUSFAR2WRT",
    "RUSFAR2W",
    "RUSFARCNRT",
    "RUSFARC1WR",
    "RUSFAR1M",
    "RUSFAR1WRT",
    "RUSFAR1W",
    "RUSFAR3M",
    "RUSFAR3MRT",
]
