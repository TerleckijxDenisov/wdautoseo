# AutoSEO Pipeline for Wildberries

Интеллектуальная система SEO-оптимизации карточек товаров на базе Apache Spark и Machine Learning.

## Быстрый старт

```bash
# Распаковать архив
unzip autoseo_pipeline.zip
cd autoseo_pipeline

# Запустить пайплайн (всё автоматически)
./run_pipeline.sh
```

Скрипт автоматически:
- Установит Java 17 (если нет)
- Создаст Python виртуальное окружение
- Установит зависимости
- Скачает датасет с HuggingFace (740 MB)
- Обработает 500K+ товаров
- Построит ML-модели
- Протестирует SEO-рекомендатор

## Системные требования

- **ОС:** macOS / Linux (Windows через WSL)
- **Python:** 3.11+
- **RAM:** 8+ GB (рекомендуется 16 GB)
- **Диск:** 5 GB свободного места
- **Java:** 17 (устанавливается автоматически)

## Структура проекта

```
autoseo_pipeline/
├── run_pipeline.sh          # Главный скрипт запуска
├── requirements.txt         # Python зависимости
├── scripts/
│   ├── parse_hf_dataset.py      # Парсинг данных с HuggingFace
│   ├── build_category_index.py  # NLP + ML пайплайн
│   ├── recommend_seo.py         # SEO-рекомендатор
│   └── scalability_experiment.py # Эксперименты
├── data/                    # Автосоздаётся
├── models/                  # Автосоздаётся
└── .venv/                   # Автосоздаётся
```

## Использование SEO-рекомендатора

После запуска пайплайна:

```bash
# Активировать окружение
source .venv/bin/activate

# Получить SEO-рекомендацию
python3 scripts/recommend_seo.py "Платье женское летнее"
python3 scripts/recommend_seo.py "Кроссовки мужские"
python3 scripts/recommend_seo.py "Сумка кожаная"
```

## Технологии

| Компонент | Технология |
|-----------|------------|
| Распределённые вычисления | Apache Spark 3.5 |
| NLP | Spark NLP 5.1 |
| ML | PySpark MLlib (TF-IDF + KMeans) |
| Источник данных | HuggingFace Datasets |
| Хранение | Parquet |

## Результаты

| Объём | Время | Скорость | Silhouette |
|-------|-------|----------|------------|
| 500K | 45 сек | 11,000+ rec/s | 0.24 |
| 1M | 86 сек | 11,600+ rec/s | 0.32 |
| 4M | 327 сек | 12,200+ rec/s | 0.20 |

## Лицензия

MIT License

## Автор

Курсовой проект, 2025
