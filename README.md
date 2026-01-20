# wdautoseo: Автоматизированный SEO-рекомендатор


Система для генерации SEO-рекомендаций на основе анализа больших данных с использованием NLP и машинного обучения.

**Требования:** Python 3.11+

### Установка Python 3.11
- **Ubuntu/WSL:** `sudo apt update && sudo apt install python3.11`
- **macOS:** `brew install python@3.11`
- **Windows:** Скачайте инсталлятор с [python.org](https://www.python.org/downloads/windows/) или используйте команду: `winget install Python.Python.3.11`

## Установка и запуск

Для полной настройки проекта и обработки данных выполните следующие команды. Скрипт `run_pipeline.sh` автоматически создаст виртуальное окружение, установит зависимости и настроит окружение Java.

> **Примечание для Windows:** Рекомендуется использовать **WSL** (Windows Subsystem for Linux) для запуска `.sh` скриптов.

```bash
# Клонирование репозитория
git clone https://github.com/TerleckijxDenisov/wdautoseo
cd wdautoseo

# Запуск пайплайна (настройка venv и обработка данных)
./run_pipeline.sh
```

## Получение SEO-рекомендаций

После завершения обработки данных вы можете получать рекомендации для конкретных запросов с помощью скрипта `recommend_seo.py`.

**Команда для запуска:**

**Linux / macOS / WSL:**
```bash
source .venv/bin/activate
python3 scripts/recommend_seo.py "Ваш поисковый запрос"
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
python scripts/recommend_seo.py "Ваш поисковый запрос"
```

*Примеры запросов:*
- `python3 scripts/recommend_seo.py "Платье женское летнее"`
- `python3 scripts/recommend_seo.py "Кроссовки мужские"`

## Технологический стек

Система базируется на технологиях распределенных вычислений и NLP:

| Компонент | Технология |
| :--- | :--- |
| Вычисления | Apache Spark 3.5 |
| NLP-обработка | Spark NLP 5.1 |
| Machine Learning | PySpark MLlib (TF-IDF + KMeans) |
| Хранение данных | Parquet, HDFS |
| Источник данных | HuggingFace Datasets |

## Производительность

Результаты тестов на различных объемах данных:

| Объём данных | Время обработки | Скорость | Silhouette Score |
| :--- | :--- | :--- | :--- |
| 500K записей | 45 сек | 11,000+ rec/s | 0.24 |
| 1M записей | 86 сек | 11,600+ rec/s | 0.32 |
| 4M записей | 327 сек | 12,200+ rec/s | 0.20 |

Для работы рекомендуется 16 GB RAM и 5 GB свободного места на диске.

Ссылка на проект: https://github.com/TerleckijxDenisov/wdautoseo

Ссылка на датасет: https://huggingface.co/datasets/nyuuzyou/wb-products/

Ссылка на демонстрацию проекта: https://drive.google.com/drive/folders/1GQzhiJddcOgZYo4vnMRQKqZvTGX1MHdN?usp=sharing
