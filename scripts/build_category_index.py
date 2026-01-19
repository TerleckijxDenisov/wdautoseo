"""
Построение индекса SEO-ключевых слов с ML-кластеризацией
ПОЛНАЯ ВЕРСИЯ - обучение на всех данных
"""
import os
import sys
import time

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
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer as SparkTokenizer
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as F
from pyspark import StorageLevel
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *


def create_spark_session():
    """Оптимизированная Spark-сессия для M3 Pro."""
    return SparkSession.builder \
        .appName("WB-Index-Builder") \
        .master("local[*]") \
        .config("spark.driver.memory", "8G") \
        .config("spark.sql.shuffle.partitions", "24") \
        .config("spark.default.parallelism", "24") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4") \
        .getOrCreate()


def build_nlp_pipeline():
    """NLP-пайплайн (оптимизированный)."""
    return Pipeline(stages=[
        DocumentAssembler().setInputCol("full_text").setOutputCol("document"),
        Tokenizer().setInputCols(["document"]).setOutputCol("token"),
        Normalizer().setInputCols(["token"]).setOutputCol("normalized").setLowercase(True),
        StopWordsCleaner.pretrained("stopwords_iso", "ru").setInputCols(["normalized"]).setOutputCol("clean"),
        Finisher().setInputCols(["clean"]).setOutputCols(["keywords"]).setOutputAsArray(True)
    ])


def build_ml_pipeline():
    """ML-пайплайн: TF-IDF + KMeans."""
    return Pipeline(stages=[
        SparkTokenizer(inputCol="text_keywords", outputCol="words"),
        HashingTF(inputCol="words", outputCol="tf", numFeatures=500),
        IDF(inputCol="tf", outputCol="features", minDocFreq=5),
        KMeans().setK(30).setSeed(42).setFeaturesCol("features").setMaxIter(10)
    ])


def main():
    total_start = time.time()
    
    print("=" * 60)
    print("ПОСТРОЕНИЕ ИНДЕКСА (ПОЛНАЯ ВЕРСИЯ)")
    print("=" * 60)
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # ===== ШАГ 1: ЗАГРУЗКА =====
    t1 = time.time()
    print("\n[1/6] Загрузка данных...")
    
    df = spark.read.csv("data/raw/real_wb_data.csv", header=True, sep='\t', inferSchema=True)
    df = df.dropDuplicates(['raw_title', 'category'])
    df = df.filter(F.col("category").isNotNull() & (F.col("category") != ""))
    df = df.withColumn("full_text", F.concat_ws(" ", F.col("raw_title"), F.col("description")))
    
    # Рапартиционируем для параллелизма и кэшируем
    df = df.repartition(24).persist(StorageLevel.MEMORY_AND_DISK)
    total = df.count()
    print(f"    Загружено: {total} записей ({time.time()-t1:.1f} сек)")
    
    # ===== ШАГ 2: NLP =====
    t2 = time.time()
    print("\n[2/6] NLP-обработка...")
    
    nlp_pipeline = build_nlp_pipeline()
    nlp_model = nlp_pipeline.fit(df.limit(1000))  # fit быстрый
    nlp_model.write().overwrite().save("models/nlp_pipeline")
    
    processed = nlp_model.transform(df)
    processed = processed.withColumn("text_keywords", F.concat_ws(" ", F.col("keywords")))
    processed = processed.select("sku_id", "category", "raw_title", "text_keywords", "keywords")
    processed = processed.repartition(24).persist(StorageLevel.MEMORY_AND_DISK)
    processed.count()  # Форсируем
    
    df.unpersist()  # Освобождаем память
    print(f"    NLP завершен ({time.time()-t2:.1f} сек)")
    
    # ===== ШАГ 3: ML-КЛАСТЕРИЗАЦИЯ =====
    t3 = time.time()
    print("\n[3/6] ML-кластеризация (TF-IDF + KMeans на ВСЕХ данных)...")
    
    ml_pipeline = build_ml_pipeline()
    ml_model = ml_pipeline.fit(processed)
    ml_model.write().overwrite().save("models/ml_pipeline")
    
    clustered = ml_model.transform(processed)
    clustered = clustered.persist(StorageLevel.MEMORY_AND_DISK)
    clustered.count()
    
    processed.unpersist()
    print(f"    ML завершен ({time.time()-t3:.1f} сек)")
    
    # ===== ШАГ 4: СОХРАНЕНИЕ ТОВАРОВ С КЛАСТЕРАМИ =====
    t4 = time.time()
    print("\n[4/6] Сохранение товаров с кластерами...")
    
    clustered.select("sku_id", "category", "raw_title", F.col("prediction").alias("cluster_id")) \
        .write.mode("overwrite").parquet("data/hdfs/clustered_products.parquet")
    print(f"    Сохранено ({time.time()-t4:.1f} сек)")
    
    # ===== ШАГ 5: КЛЮЧЕВЫЕ СЛОВА ПО КЛАСТЕРАМ =====
    t5 = time.time()
    print("\n[5/6] Ключевые слова по кластерам...")
    
    from pyspark.sql.window import Window
    
    cluster_kw = clustered.select("prediction", F.explode("keywords").alias("word"))
    cluster_kw = cluster_kw.filter(F.length("word") > 2)
    cluster_counts = cluster_kw.groupBy("prediction", "word").count()
    
    w = Window.partitionBy("prediction").orderBy(F.desc("count"))
    cluster_top = cluster_counts.withColumn("r", F.row_number().over(w)).filter(F.col("r") <= 10)
    cluster_top = cluster_top.groupBy("prediction").agg(F.collect_list("word").alias("cluster_keywords"))
    cluster_top.withColumnRenamed("prediction", "cluster_id") \
        .write.mode("overwrite").parquet("data/hdfs/cluster_keywords.parquet")
    print(f"    Кластеры сохранены ({time.time()-t5:.1f} сек)")
    
    # ===== ШАГ 6: КЛЮЧЕВЫЕ СЛОВА ПО КАТЕГОРИЯМ =====
    t6 = time.time()
    print("\n[6/6] Ключевые слова по категориям...")
    
    cat_kw = clustered.select("category", F.explode("keywords").alias("word"))
    cat_kw = cat_kw.filter(F.length("word") > 2)
    cat_counts = cat_kw.groupBy("category", "word").count()
    
    w2 = Window.partitionBy("category").orderBy(F.desc("count"))
    cat_top = cat_counts.withColumn("r", F.row_number().over(w2)).filter(F.col("r") <= 10)
    cat_top = cat_top.groupBy("category").agg(
        F.collect_list("word").alias("top_keywords"),
        F.sum("count").alias("total_keyword_count")
    )
    
    cat_counts_df = clustered.groupBy("category").count().withColumnRenamed("count", "product_count")
    final_index = cat_top.join(cat_counts_df, "category")
    final_index.write.mode("overwrite").parquet("data/hdfs/category_keywords.parquet")
    print(f"    Категории сохранены ({time.time()-t6:.1f} сек)")
    
    # ===== ИТОГ =====
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"ГОТОВО! Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"Обработано: {total} товаров")
    print(f"Категорий: {final_index.count()}")
    print(f"Кластеров: 30")
    print("=" * 60)
    
    final_index.orderBy(F.desc("product_count")).show(5, truncate=50)
    
    spark.stop()


if __name__ == "__main__":
    main()
