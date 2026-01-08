# -*- coding: utf-8 -*-
"""
Simulation Hive avec SparkSQL - MovieLens Data Analysis
Ce script simule l'exécution des requêtes Hive en utilisant SparkSQL
"""

import os
# Forcer l'utilisation de Java 11 (compatible avec Spark)
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

# Configurer Hadoop pour Windows
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession

# ============================================================================
# Initialisation de SparkSession avec support Hive
# ============================================================================

print("=" * 60)
print("SIMULATION HIVE - Initialisation de SparkSession")
print("=" * 60)

spark = SparkSession.builder \
    .appName("Hive Simulation - MovieLens") \
    .master("local[*]") \
    .config("spark.sql.warehouse.dir", "spark-warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# ============================================================================
# EXTRACT - Création des tables externes (Bronze)
# ============================================================================

print("\n" + "=" * 60)
print("EXTRACT - Création des tables externes")
print("=" * 60)

# Supprimer les tables si elles existent
spark.sql("DROP TABLE IF EXISTS movies_raw")
spark.sql("DROP TABLE IF EXISTS ratings_raw")
spark.sql("DROP TABLE IF EXISTS silver_movielens")

# ----------------------------------------------------------------------------
# [TABLE 1] Table externe pour movies.csv
# Code HQL équivalent: CREATE EXTERNAL TABLE movies_raw (...)
# ----------------------------------------------------------------------------
print("\n[TABLE 1] Création de la table 'movies_raw'...")
movies_df = spark.read.option("header", True).csv("ml-latest/movies.csv")
movies_df.createOrReplaceTempView("movies_raw")

print("Schema de movies_raw:")
spark.sql("DESCRIBE movies_raw").show()

# ============================================================================
# AFFICHAGE 1: Vérification des données - movies_raw
# Code: SELECT * FROM movies_raw LIMIT 5
# ============================================================================
print("\n[AFFICHAGE 1] Contenu de movies_raw (LIMIT 5):")
spark.sql("SELECT * FROM movies_raw LIMIT 5").show(truncate=False)

# ----------------------------------------------------------------------------
# [TABLE 2] Table externe pour ratings.csv
# Code HQL équivalent: CREATE EXTERNAL TABLE ratings_raw (...)
# ----------------------------------------------------------------------------
print("\n[TABLE 2] Création de la table 'ratings_raw'...")
ratings_df = spark.read.option("header", True).csv("ml-latest/ratings.csv")
ratings_df.createOrReplaceTempView("ratings_raw")

print("Schema de ratings_raw:")
spark.sql("DESCRIBE ratings_raw").show()

# ============================================================================
# AFFICHAGE 2: Vérification des données - ratings_raw
# Code: SELECT * FROM ratings_raw LIMIT 5
# ============================================================================
print("\n[AFFICHAGE 2] Contenu de ratings_raw (LIMIT 5):")
spark.sql("SELECT * FROM ratings_raw LIMIT 5").show()

# ============================================================================
# TRANSFORM - Création de la table Silver
# ============================================================================

print("\n" + "=" * 60)
print("TRANSFORM - Création de la table Silver")
print("=" * 60)

# ----------------------------------------------------------------------------
# [TABLE 3] Table Silver avec transformations
# Code HQL équivalent: CREATE TABLE silver_movielens AS SELECT ...
# Transformations:
#   - Extraction de l'année avec regexp_extract
#   - Explosion des genres avec LATERAL VIEW explode
#   - Jointure avec les statistiques de ratings
# ----------------------------------------------------------------------------
print("\n[TABLE 3] Création de la table 'silver_movielens'...")

# Étape 1: Créer la table des statistiques de ratings
spark.sql("""
    CREATE OR REPLACE TEMP VIEW ratings_stats AS
    SELECT
        movieId,
        COUNT(rating) AS rating_count,
        AVG(CAST(rating AS FLOAT)) AS rating_avg
    FROM ratings_raw
    GROUP BY movieId
""")

# Étape 2: Créer la table silver avec explosion des genres
spark.sql("""
    CREATE OR REPLACE TEMP VIEW silver_movielens AS
    SELECT
        CAST(m.movieId AS INT) AS movie_id,
        m.title AS movie_name,
        CAST(regexp_extract(m.title, '\\\\((\\\\d{4})\\\\)', 1) AS INT) AS year,
        genre,
        CAST(r.rating_count AS INT) AS rating_count,
        r.rating_avg
    FROM movies_raw m
    JOIN ratings_stats r ON m.movieId = r.movieId
    LATERAL VIEW explode(split(m.genres, '\\\\|')) exploded_genres AS genre
""")

print("Schema de silver_movielens:")
spark.sql("DESCRIBE silver_movielens").show()

# ============================================================================
# AFFICHAGE 3: Vérification de la table Silver
# Code: SELECT * FROM silver_movielens LIMIT 10
# ============================================================================
print("\n[AFFICHAGE 3] Contenu de silver_movielens (LIMIT 10):")
spark.sql("SELECT * FROM silver_movielens LIMIT 10").show(truncate=False)

# ============================================================================
# DATA ANALYSIS - Requêtes HiveQL
# ============================================================================

print("\n" + "=" * 60)
print("DATA ANALYSIS - Requêtes HiveQL")
print("=" * 60)

# ============================================================================
# AFFICHAGE 4: Meilleur film par année
# Code HQL: SELECT year, movie_name, rating_avg, rating_count
#           FROM (SELECT ..., ROW_NUMBER() OVER (...) AS rank FROM silver_movielens)
#           WHERE rank = 1 ORDER BY year DESC
# ============================================================================
print("\n[AFFICHAGE 4] Meilleur film par année:")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT
            year,
            movie_name,
            rating_avg,
            rating_count,
            ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) AS rank
        FROM silver_movielens
        WHERE year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================================
# AFFICHAGE 5: Meilleur film par genre
# Code HQL: SELECT genre, movie_name, rating_avg, rating_count
#           FROM (SELECT ..., ROW_NUMBER() OVER (...) AS rank FROM silver_movielens)
#           WHERE rank = 1 ORDER BY genre
# ============================================================================
print("\n[AFFICHAGE 5] Meilleur film par genre:")
spark.sql("""
    SELECT genre, movie_name, rating_avg, rating_count
    FROM (
        SELECT
            genre,
            movie_name,
            rating_avg,
            rating_count,
            ROW_NUMBER() OVER (PARTITION BY genre ORDER BY rating_avg DESC, rating_count DESC) AS rank
        FROM silver_movielens
    ) ranked
    WHERE rank = 1
    ORDER BY genre
""").show(20, truncate=False)

# ============================================================================
# AFFICHAGE 6: Meilleur film 'Action' par année
# Code HQL: SELECT ... WHERE genre = 'Action' ...
# ============================================================================
print("\n[AFFICHAGE 6] Meilleur film 'Action' par année:")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT
            year,
            movie_name,
            rating_avg,
            rating_count,
            ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) AS rank
        FROM silver_movielens
        WHERE genre = 'Action' AND year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================================
# AFFICHAGE 7: Meilleur film 'Romance' par année
# Code HQL: SELECT ... WHERE genre = 'Romance' ...
# ============================================================================
print("\n[AFFICHAGE 7] Meilleur film 'Romance' par année:")
spark.sql("""
    SELECT year, movie_name, rating_avg, rating_count
    FROM (
        SELECT
            year,
            movie_name,
            rating_avg,
            rating_count,
            ROW_NUMBER() OVER (PARTITION BY year ORDER BY rating_avg DESC, rating_count DESC) AS rank
        FROM silver_movielens
        WHERE genre = 'Romance' AND year IS NOT NULL
    ) ranked
    WHERE rank = 1
    ORDER BY year DESC
""").show(20, truncate=False)

# ============================================================================
# FIN
# ============================================================================
print("\n" + "=" * 60)
print("Simulation Hive terminée avec succès!")
print("=" * 60)

spark.stop()
