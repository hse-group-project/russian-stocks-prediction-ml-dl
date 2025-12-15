#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "Этот скрипт должен запускаться от имени root" >&2
  exit 1
fi

set -e

BACKUP_DIR="./backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql"

DB_NAME="russian-stocks-prediction-ml-dl"
DB_USER="root"
DB_SERVICE="db"

echo "Создание резервной копии в $BACKUP_FILE..."

# Без кавычек вокруг DB_NAME!
docker-compose exec -T "$DB_SERVICE" pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"

# Проверка на пустоту
if [ ! -s "$BACKUP_FILE" ]; then
  echo "❌ ОШИБКА: резервная копия пуста!" >&2
  exit 1
fi

echo "✅ Успешно: $BACKUP_FILE"
