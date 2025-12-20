Все команды выполняем в корневой директории
1. Инициализация окружения: `uv sync`

2. Запуск backend: `uv run python -m uvicorn src.api.main:app  --port 8000 --reload`

3. Запуск frontend: `uv run streamlit run src/web/main.py`