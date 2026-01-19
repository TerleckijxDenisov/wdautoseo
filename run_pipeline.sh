#!/bin/bash
# ===========================================================================
# AutoSEO Pipeline - Полный автоматический запуск
# ===========================================================================
# Этот скрипт выполняет весь пайплайн за один запуск:
# 1. Создает необходимые директории
# 2. Скачивает шард с HuggingFace (если нет)
# 3. Парсит данные и удаляет дубликаты (500к строк)
# 4. Строит индекс с ML-кластеризацией
# 5. Тестирует SEO-рекомендатор
# ===========================================================================

set -e  # Останавливаемся при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Корневая директория проекта
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ===========================================================================
# Проверка и установка Java 17 (требуется для Spark)
# ===========================================================================
JAVA_DIR="$PROJECT_DIR/java"
JAVA_HOME_PATH="$JAVA_DIR/jdk-17.0.2.jdk/Contents/Home"

if [ -d "$JAVA_HOME_PATH" ]; then
    echo -e "${GREEN}✓ Java 17 найдена${NC}"
elif command -v java &> /dev/null && java -version 2>&1 | grep -q "17"; then
    echo -e "${GREEN}✓ Системная Java 17 найдена${NC}"
    JAVA_HOME_PATH=$(dirname $(dirname $(readlink -f $(which java))))
else
    echo -e "${YELLOW}Java 17 не найдена. Установка...${NC}"
    
    mkdir -p "$JAVA_DIR"
    cd "$JAVA_DIR"
    
    # Определяем архитектуру
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        JDK_URL="https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_macos-aarch64_bin.tar.gz"
    else
        JDK_URL="https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_macos-x64_bin.tar.gz"
    fi
    
    echo "Скачивание OpenJDK 17..."
    curl -L -o openjdk.tar.gz "$JDK_URL" --progress-bar
    
    echo "Распаковка..."
    tar -xzf openjdk.tar.gz
    rm openjdk.tar.gz
    
    cd "$PROJECT_DIR"
    echo -e "${GREEN}✓ Java 17 установлена${NC}"
fi

export JAVA_HOME="$JAVA_HOME_PATH"
export PATH="$JAVA_HOME/bin:$PATH"

# ===========================================================================
# Python виртуальное окружение и зависимости
# ===========================================================================
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Создание виртуального окружения...${NC}"
    python3 -m venv .venv
fi

echo -e "${BLUE}Активация виртуального окружения...${NC}"
source .venv/bin/activate

# Установка зависимостей
echo -e "${BLUE}Проверка/установка зависимостей...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Зависимости установлены${NC}"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           AutoSEO Pipeline - Wildberries                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ===========================================================================
# ШАГ 1: Создание директорий
# ===========================================================================
echo -e "${YELLOW}[1/5] Создание директорий...${NC}"

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/hdfs
mkdir -p models

echo -e "${GREEN}✓ Директории созданы${NC}"
echo ""

# ===========================================================================
# ШАГ 2: Скачивание датасета с HuggingFace
# ===========================================================================
echo -e "${YELLOW}[2/5] Проверка датасета...${NC}"

SHARD_FILE="data/raw/basket-01.json.zst"
HF_URL="https://huggingface.co/datasets/nyuuzyou/wb-products/resolve/main/basket-01.json.zst"

if [ -f "$SHARD_FILE" ]; then
    echo -e "${GREEN}✓ Датасет уже скачан: $SHARD_FILE${NC}"
else
    echo -e "${BLUE}Скачивание шарда с HuggingFace...${NC}"
    echo "URL: $HF_URL"
    
    # Проверяем наличие curl или wget
    if command -v curl &> /dev/null; then
        curl -L -o "$SHARD_FILE" "$HF_URL" --progress-bar
    elif command -v wget &> /dev/null; then
        wget -O "$SHARD_FILE" "$HF_URL" --show-progress
    else
        echo -e "${RED}Ошибка: установите curl или wget${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Датасет скачан${NC}"
fi
echo ""

# ===========================================================================
# ШАГ 3: Парсинг данных (500к строк после дедупликации)
# ===========================================================================
echo -e "${YELLOW}[3/5] Парсинг датасета...${NC}"

CSV_FILE="data/raw/real_wb_data.csv"

# Проверяем, нужно ли парсить заново
if [ -f "$CSV_FILE" ]; then
    LINE_COUNT=$(wc -l < "$CSV_FILE" | tr -d ' ')
    echo "Существующий файл: $LINE_COUNT строк"
    
    if [ "$LINE_COUNT" -ge 500000 ]; then
        echo -e "${GREEN}✓ CSV уже содержит достаточно данных${NC}"
    else
        echo -e "${BLUE}Перепарсинг для получения 500к строк...${NC}"
        python3 scripts/parse_hf_dataset.py --input "$SHARD_FILE" --output "$CSV_FILE" --limit 1300000
    fi
else
    echo -e "${BLUE}Парсинг .zst файла -> CSV (лимит 1.3М для 500к после дедупликации)...${NC}"
    python3 scripts/parse_hf_dataset.py --input "$SHARD_FILE" --output "$CSV_FILE" --limit 1300000
fi

echo -e "${GREEN}✓ Данные готовы${NC}"
echo ""

# ===========================================================================
# ШАГ 4: Построение индекса с ML-кластеризацией
# ===========================================================================
echo -e "${YELLOW}[4/5] Построение индекса (NLP + TF-IDF + KMeans)...${NC}"

# Проверяем, есть ли уже готовые модели
if [ -d "models/ml_pipeline" ] && [ -d "data/hdfs/clustered_products.parquet" ]; then
    echo "Модели уже существуют."
    read -p "Пересобрать индекс? (y/n): " REBUILD
    if [ "$REBUILD" = "y" ] || [ "$REBUILD" = "Y" ]; then
        python3 scripts/build_category_index.py
    else
        echo -e "${GREEN}✓ Используем существующие модели${NC}"
    fi
else
    python3 scripts/build_category_index.py
fi

echo -e "${GREEN}✓ Индекс построен${NC}"
echo ""

# ===========================================================================
# ШАГ 5: Тестирование SEO-рекомендатора
# ===========================================================================
echo -e "${YELLOW}[5/5] Тестирование SEO-рекомендатора...${NC}"

echo -e "${BLUE}Тест 1: Платье женское${NC}"
python3 scripts/recommend_seo.py "Платье женское летнее"

echo -e "${BLUE}Тест 2: Кроссовки мужские${NC}"
python3 scripts/recommend_seo.py "Кроссовки мужские беговые"

echo -e "${BLUE}Тест 3: Сумка кожаная${NC}"
python3 scripts/recommend_seo.py "Сумка кожаная женская"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ПАЙПЛАЙН ЗАВЕРШЕН!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Результаты:"
echo "  • CSV данные:        data/raw/real_wb_data.csv"
echo "  • NLP модель:        models/nlp_pipeline/"
echo "  • ML модель:         models/ml_pipeline/"
echo "  • Кластеры товаров:  data/hdfs/clustered_products.parquet"
echo "  • Кластеры KW:       data/hdfs/cluster_keywords.parquet"
echo "  • Категории KW:      data/hdfs/category_keywords.parquet"
echo ""
echo "Для рекомендаций запустите:"
echo "  python3 scripts/recommend_seo.py \"ваш товар\""
echo ""
