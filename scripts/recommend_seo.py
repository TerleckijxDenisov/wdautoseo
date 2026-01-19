"""
SEO-рекомендатор с ML-кластеризацией

ML-модель (TF-IDF + KMeans) уже применена ко всем товарам.
Этот скрипт использует результаты кластеризации из parquet-файлов.

Использование: python3 scripts/recommend_seo.py "название товара"
"""
import sys
import os
import random

# Определяем JAVA_HOME (из окружения или стандартные пути)
if not os.environ.get("JAVA_HOME"):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    java_paths = [
        os.path.join(project_dir, "java/jdk-17.0.2.jdk/Contents/Home"),
        os.path.expanduser("~/java-arm64/jdk-17.0.2.jdk/Contents/Home"),
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    ]
    for path in java_paths:
        if os.path.exists(path):
            os.environ["JAVA_HOME"] = path
            break

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def simple_stem_ru(word: str) -> str:
    """Простой стеммер для русского языка."""
    word = word.lower()
    endings = ['ами', 'ями', 'ях', 'ий', 'ый', 'ая', 'ое', 'ые', 'ие', 
               'ей', 'ой', 'ом', 'ем', 'ую', 'юю', 'ов', 'ев', 'ах', 
               'ья', 'ье', 'ьи', 'а', 'я', 'ы', 'и', 'у', 'ю', 'о', 'е', 'ь']
    for ending in endings:
        if len(word) > len(ending) + 2 and word.endswith(ending):
            return word[:-len(ending)]
    return word


def find_best_category(user_input: str, categories_df):
    """Находит наиболее подходящую категорию."""
    user_words = set(user_input.lower().split())
    categories = categories_df.collect()
    best_match, best_score = None, 0
    
    for row in categories:
        category_name = row['category'].lower()
        category_words = category_name.split()
        score = 0
        
        for word in user_words:
            word_stem = simple_stem_ru(word)
            if category_words and simple_stem_ru(category_words[0]) == word_stem:
                score += 10
            for cat_word in category_words:
                if simple_stem_ru(cat_word) == word_stem:
                    score += 3
        
        score += min(row['product_count'] / 1000, 5)
        score -= len(category_words) * 0.5
        
        if score > best_score:
            best_score, best_match = score, row
    
    return best_match, best_score


def find_cluster_by_category(spark, category_name, user_input):
    """
    Находит ML-кластер на основе похожих товаров из той же категории.
    Использует результаты работы KMeans (clustered_products.parquet).
    """
    if not os.path.exists("data/hdfs/clustered_products.parquet"):
        return None
    
    products = spark.read.parquet("data/hdfs/clustered_products.parquet")
    category_products = products.filter(F.col("category") == category_name)
    
    # Ищем товар с похожим названием
    for word in user_input.lower().split():
        if len(word) > 3:
            match = category_products.filter(
                F.lower(F.col("raw_title")).contains(word)
            ).first()
            if match:
                return match['cluster_id']
    
    # Иначе берем самый популярный кластер в категории
    top_cluster = category_products.groupBy("cluster_id").count() \
        .orderBy(F.desc("count")).first()
    return top_cluster['cluster_id'] if top_cluster else None


def get_cluster_keywords(spark, cluster_id):
    """Получает ключевые слова ML-кластера."""
    if not os.path.exists("data/hdfs/cluster_keywords.parquet"):
        return []
    clusters = spark.read.parquet("data/hdfs/cluster_keywords.parquet")
    row = clusters.filter(F.col("cluster_id") == cluster_id).first()
    return list(row['cluster_keywords']) if row else []


def predict_seo(product_description: str):
    """Основная функция генерации SEO."""
    spark = SparkSession.builder \
        .appName("WB-SEO-Recommender") \
        .master("local[*]") \
        .config("spark.driver.memory", "4G") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    if not os.path.exists("data/hdfs/category_keywords.parquet"):
        print("Ошибка: Запустите build_category_index.py")
        spark.stop()
        return
    
    categories_df = spark.read.parquet("data/hdfs/category_keywords.parquet")
    
    # Шаг 1: Категория
    match, _ = find_best_category(product_description, categories_df)
    if not match:
        match = categories_df.orderBy(F.desc("product_count")).first()
    
    category_name = match['category']
    category_keywords = list(match['top_keywords'])
    product_count = match['product_count']
    
    # Шаг 2: ML-кластер (из результатов KMeans)
    cluster_id = find_cluster_by_category(spark, category_name, product_description)
    cluster_keywords = get_cluster_keywords(spark, cluster_id) if cluster_id else []
    
    # Шаг 3: Объединение ключевых слов
    all_keywords = category_keywords.copy()
    for kw in cluster_keywords:
        if kw not in all_keywords:
            all_keywords.append(kw)
    
    filtered = [w for w in all_keywords if w.lower() not in category_name.lower()][:8]
    
    # Шаг 4: SEO-генерация
    base = product_description.strip().capitalize()
    selected = random.sample(filtered, min(3, len(filtered))) if filtered else category_keywords[:3]
    
    adjectives = [w for w in filtered if w.endswith(('ый', 'ий', 'ая', 'ое', 'ые', 'ие'))]
    
    if adjectives:
        title = f"{adjectives[0].capitalize()} {base} — {' '.join(selected)} | {category_name}"
    else:
        title = f"{base} — {' '.join(selected)} | {category_name}"
    
    if cluster_id is not None:
        desc = f"{base} из категории '{category_name}'. ML-кластеризация (KMeans): кластер #{cluster_id}. Ключевые слова: {', '.join(all_keywords[:5])}."
    else:
        desc = f"{base} из категории '{category_name}' ({product_count} товаров). Характеристики: {', '.join(category_keywords[:4])}."
    
    # Вывод
    print("\n" + "=" * 70)
    print(f"ВХОДНЫЕ ДАННЫЕ: {product_description}")
    print(f"КАТЕГОРИЯ: {category_name} ({product_count} товаров)")
    if cluster_id is not None:
        print(f"ML-КЛАСТЕР (KMeans): #{cluster_id}")
    print("-" * 70)
    print(f"КЛЮЧЕВЫЕ СЛОВА КАТЕГОРИИ: {', '.join(category_keywords[:5])}")
    if cluster_keywords:
        print(f"КЛЮЧЕВЫЕ СЛОВА КЛАСТЕРА: {', '.join(cluster_keywords[:5])}")
    print("-" * 70)
    print(f"SEO-ЗАГОЛОВОК:\n  {title}")
    print("-" * 70)
    print(f"SEO-ОПИСАНИЕ:\n  {desc}")
    print("=" * 70 + "\n")
    
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Введите описание товара: ")
    predict_seo(user_input)
