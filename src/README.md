#### Все команды выполняем в корневой директории!

1. Создание .env: создайте .env в корне проекта со следующим содержимым:

```
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
DATABASE_URL=postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}
ADMIN_DELETE_TOKEN=
```

*важно! миграцию проводить только для своей бд, ADMIN_DELETE_TOKEN уникальный для данного проекта, оригинал только у авторов*

2. Инициализация окружения: `uv sync`

3. Включить окружение: `source .venv/bin/activate` - macos, `.venv\Scripts\activate` - windows

4. Запуск backend: `uv run python -m uvicorn src.api.main:app --port 8000 --reload`

5. Запуск frontend: `uv run streamlit run src/web/main.py`