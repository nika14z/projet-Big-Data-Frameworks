# -*- coding: utf-8 -*-
"""
Spark ETL Pipeline - MovieLens Data Analysis
"""

import os
# Forcer l'utilisation de Java 11 (compatible avec Spark)
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

# Configurer Hadoop pour Windows (nécessaire pour écrire des fichiers Parquet)
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, split, explode, avg, count, col, row_number, desc
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window

# ============================================================================
# EXTRACT - Initialisation et chargement des données
# ============================================================================

# Initialisation de la session Spark
# Code: lignes 25-28
spark = SparkSession.builder \
    .appName("MovieLens ETL") \
    .master("local[*]") \
    .getOrCreate()

# Charger les fichiers CSV
print("=" * 60)
print("EXTRACT - Chargement des fichiers CSV")
print("=" * 60)

# Code: lignes 35-36 - Chargement des CSV
movies_df = spark.read.option("header", True).csv("ml-latest/movies.csv")
ratings_df = spark.read.option("header", True).csv("ml-latest/ratings.csv")

# Code: lignes 39-42 - Conversion des types de colonnes
ratings_df = ratings_df.withColumn("movieId", col("movieId").cast(IntegerType())) \
                       .withColumn("rating", col("rating").cast(FloatType()))

movies_df = movies_df.withColumn("movieId", col("movieId").cast(IntegerType()))

# ============================================================
# AFFICHAGE 1: Premières lignes de movies_df
# Code: ligne 48 - movies_df.show(5, truncate=False)
# Code: ligne 49 - movies_df.printSchema()
# ============================================================
print("\n[AFFICHAGE 1] Premières lignes de 'movies_df' (ligne 48-49):")
movies_df.show(5, truncate=False)
movies_df.printSchema()

# ============================================================
# AFFICHAGE 2: Premières lignes de ratings_df
# Code: ligne 55 - ratings_df.show(5)
# Code: ligne 56 - ratings_df.printSchema()
# ============================================================
print("\n[AFFICHAGE 2] Premières lignes de 'ratings_df' (ligne 55-56):")
ratings_df.show(5)
ratings_df.printSchema()

# ============================================================================
# TRANSFORM - Traitement et transformations
# ============================================================================

print("\n" + "=" * 60)
print("TRANSFORM - Traitement des données")
print("=" * 60)

# Code: lignes 70-78 - Extraction de l'année de sortie
print("\n1. Extraction de l'année de sortie des films...")
movies_df = movies_df.withColumn(
    "year",
    regexp_extract("title", r"\((\d{4})\)", 1)
)

# Gérer les années manquantes (remplacer les chaînes vides par null)
movies_df = movies_df.withColumn(
    "year",
    col("year").cast(IntegerType())
)

# ============================================================
# AFFICHAGE 3: DataFrame après extraction de l'année
# Code: ligne 84 - movies_df.show(5, truncate=False)
# ============================================================
print("\n[AFFICHAGE 3] Après extraction de l'année (ligne 84):")
movies_df.show(5, truncate=False)

# Code: lignes 89-93 - Explosion des genres (un genre par ligne)
print("\n2. Explosion des genres (un genre par ligne)...")
movies_exploded_df = movies_df.withColumn(
    "genre",
    explode(split("genres", r"\|"))
)

# ============================================================
# AFFICHAGE 4: DataFrame après explosion des genres
# Code: ligne 98 - movies_exploded_df.show(10, truncate=False)
# ============================================================
print("\n[AFFICHAGE 4] Après explosion des genres (ligne 98):")
movies_exploded_df.show(10, truncate=False)

# Code: lignes 103-109 - Calcul des statistiques de ratings
print("\n3. Calcul des statistiques de ratings...")
ratings_stats = ratings_df.groupBy("movieId") \
    .agg(
        count("rating").alias("rating_count"),
        avg("rating").alias("rating_avg")
    )

# ============================================================
# AFFICHAGE 5: Statistiques de ratings par film
# Code: ligne 114 - ratings_stats.show(5)
# ============================================================
print("\n[AFFICHAGE 5] Statistiques de ratings (ligne 114):")
ratings_stats.show(5)

# Code: lignes 119-130 - Jointure entre movies et ratings
print("\n4. Jointure entre movies et ratings...")
final_df = movies_exploded_df.join(
    ratings_stats,
    movies_exploded_df.movieId == ratings_stats.movieId,
    "inner"
).select(
    movies_exploded_df.movieId.alias("movie_id"),
    movies_exploded_df.title.alias("movie_name"),
    "year",
    "genre",
    "rating_count",
    "rating_avg"
)

# ============================================================
# AFFICHAGE 6: Résultat final après jointure (dataset silver)
# Code: ligne 135 - final_df.show(10, truncate=False)
# ============================================================
print("\n[AFFICHAGE 6] Résultat après jointure - Dataset Silver (ligne 135):")
final_df.show(10, truncate=False)

# ============================================================================
# LOAD - Sauvegarde des données transformées
# ============================================================================

print("\n" + "=" * 60)
print("LOAD - Sauvegarde dans Parquet")
print("=" * 60)

# Code: ligne 149 - Sauvegarde en Parquet
print("\nSauvegarde des résultats dans 'silver_movielens' (Parquet)...")
final_df.write.mode("overwrite").parquet("silver_movielens")
print("Sauvegarde terminée!")

# ============================================================================
# DATA ANALYSIS - Analyse avec Spark SQL
# ============================================================================

print("\n" + "=" * 60)
print("DATA ANALYSIS - Requêtes Spark SQL")
print("=" * 60)

# Code: ligne 162 - Chargement du Parquet
print("\nChargement du dataset 'silver_movielens'...")
silver_df = spark.read.parquet("silver_movielens")

# Code: ligne 166 - Création de la vue temporaire
silver_df.createOrReplaceTempView("movies")

# ============================================================
# AFFICHAGE 7: Meilleur film par année (SQL)
# Code: lignes 172-186 - Requête SQL avec ROW_NUMBER
# ============================================================
print("\n[AFFICHAGE 7] Meilleur film par année - SQL (lignes 172-186):")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT year, movie_name, rating_avg, rating_count,
               ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) as rank
        FROM movies
        WHERE year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================
# AFFICHAGE 8: Meilleur film par genre (SQL)
# Code: lignes 192-204 - Requête SQL avec ROW_NUMBER
# ============================================================
print("\n[AFFICHAGE 8] Meilleur film par genre - SQL (lignes 192-204):")
spark.sql("""
    SELECT genre, movie_name, rating_avg, rating_count
    FROM (
        SELECT genre, movie_name, rating_avg, rating_count,
               ROW_NUMBER() OVER (PARTITION BY genre ORDER BY rating_avg DESC, rating_count DESC) as rank
        FROM movies
    ) ranked
    WHERE rank = 1
    ORDER BY genre
""").show(20, truncate=False)

# ============================================================
# AFFICHAGE 9: Meilleur film 'Action' par année (SQL)
# Code: lignes 210-223 - Requête SQL filtrée sur Action
# ============================================================
print("\n[AFFICHAGE 9] Meilleur film 'Action' par année - SQL (lignes 210-223):")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT year, movie_name, rating_avg, rating_count,
               ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) as rank
        FROM movies
        WHERE genre = 'Action' AND year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================
# AFFICHAGE 10: Meilleur film 'Romance' par année (SQL)
# Code: lignes 229-242 - Requête SQL filtrée sur Romance
# ============================================================
print("\n[AFFICHAGE 10] Meilleur film 'Romance' par année - SQL (lignes 229-242):")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT year, movie_name, rating_avg, rating_count,
               ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) as rank
        FROM movies
        WHERE genre = 'Romance' AND year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================================
# DATA ANALYSIS - Avec Window Functions (DataFrame API)
# ============================================================================

print("\n" + "=" * 60)
print("DATA ANALYSIS - Avec Window Functions (DataFrame API)")
print("=" * 60)

# Code: lignes 253-254 - Définition des fenêtres de partitionnement
window_year = Window.partitionBy("year").orderBy(desc("rating_avg"), desc("rating_count"))
window_genre = Window.partitionBy("genre").orderBy(desc("rating_avg"), desc("rating_count"))

# ============================================================
# AFFICHAGE 11: Meilleur film par année (DataFrame API)
# Code: lignes 260-267 - Utilisation de Window + row_number()
# ============================================================
print("\n[AFFICHAGE 11] Meilleur film par année - DataFrame API (lignes 260-267):")
best_movie_per_year = silver_df \
    .filter(col("year").isNotNull()) \
    .withColumn("rank", row_number().over(window_year)) \
    .filter(col("rank") == 1) \
    .select("year", "movie_name", "rating_avg", "rating_count") \
    .orderBy(desc("year"))

best_movie_per_year.show(20, truncate=False)

# ============================================================
# AFFICHAGE 12: Meilleur film par genre (DataFrame API)
# Code: lignes 274-280 - Utilisation de Window + row_number()
# ============================================================
print("\n[AFFICHAGE 12] Meilleur film par genre - DataFrame API (lignes 274-280):")
best_movie_per_genre = silver_df \
    .withColumn("rank", row_number().over(window_genre)) \
    .filter(col("rank") == 1) \
    .select("genre", "movie_name", "rating_avg", "rating_count") \
    .orderBy("genre")

best_movie_per_genre.show(20, truncate=False)

# ============================================================
# AFFICHAGE 13: Meilleur film 'Action' par année (DataFrame API)
# Code: lignes 287-296 - Filtrage sur Action + Window
# ============================================================
print("\n[AFFICHAGE 13] Meilleur film 'Action' par année - DataFrame API (lignes 287-296):")
action_df = silver_df.filter(col("genre") == "Action")
window_action = Window.partitionBy("year").orderBy(desc("rating_avg"), desc("rating_count"))

best_action_per_year = action_df \
    .filter(col("year").isNotNull()) \
    .withColumn("rank", row_number().over(window_action)) \
    .filter(col("rank") == 1) \
    .select("year", "movie_name", "rating_avg", "rating_count") \
    .orderBy(desc("year"))

best_action_per_year.show(20, truncate=False)

# ============================================================
# AFFICHAGE 14: Meilleur film 'Romance' par année (DataFrame API)
# Code: lignes 303-312 - Filtrage sur Romance + Window
# ============================================================
print("\n[AFFICHAGE 14] Meilleur film 'Romance' par année - DataFrame API (lignes 303-312):")
romance_df = silver_df.filter(col("genre") == "Romance")
window_romance = Window.partitionBy("year").orderBy(desc("rating_avg"), desc("rating_count"))

best_romance_per_year = romance_df \
    .filter(col("year").isNotNull()) \
    .withColumn("rank", row_number().over(window_romance)) \
    .filter(col("rank") == 1) \
    .select("year", "movie_name", "rating_avg", "rating_count") \
    .orderBy(desc("year"))

best_romance_per_year.show(20, truncate=False)

# Arrêter la session Spark
print("\n" + "=" * 60)
print("ETL Pipeline terminé avec succès!")
print("=" * 60)
spark.stop()
