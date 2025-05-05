import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, year, to_date
from pyspark.sql.types import IntegerType, DateType
from pyspark import SparkContext



variables = [
    'JAVA_HOME',
    'HADOOP_HOME',
    'SPARK_HOME',
    'SPARK_DIST_CLASSPATH',
    'PYSPARK_PYTHON',
    'PYSPARK_DRIVER_PYTHON'
]

print('variaveis de ambiente:')
for var in variables:
    value = os.environ.get(var, 'nao definida')
    print(f'{var}={value}')

#inicar a sessao spark

spark = SparkSession.builder.getOrCreate()
print(f'versao do spark:{spark.version}')
# Inicializar a sessão Spark
spark = SparkSession.builder \
    .appName("Tratamento de Dados de Vídeos") \
    .getOrCreate()

# 1. Ler o arquivo 'videos-stats.csv' no dataframe 'df_video'
df_video = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("videos-stats.csv")

# 2. Alterar valores nulos para 0 nos campos 'Likes', 'Comments' e 'Views'
df_video = df_video.fillna(0, subset=['Likes', 'Comments', 'Views'])

# 3. Ler o arquivo 'comments.csv' no dataframe 'df_comentario'
df_comentario = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("comments.csv")

# 4. Calcular a quantidade de registros dos dataframes
print("Quantidade de registros em df_video:", df_video.count())
print("Quantidade de registros em df_comentario:", df_comentario.count())

# 5. Remover registros com 'Video ID' nulos e recalcular a quantidade
df_video = df_video.filter(col("Video ID").isNotNull())
df_comentario = df_comentario.filter(col("Video ID").isNotNull())

print("Quantidade após remover nulos em df_video:", df_video.count())
print("Quantidade após remover nulos em df_comentario:", df_comentario.count())

# 6. Remover registros com 'Video ID' duplicados apenas no df_video
df_video = df_video.dropDuplicates(["Video ID"])

# 7. Converter campos para int no df_video
df_video = df_video.withColumn("Likes", col("Likes").cast(IntegerType())) \
    .withColumn("Comments", col("Comments").cast(IntegerType())) \
    .withColumn("Views", col("Views").cast(IntegerType()))

# 8. Converter campos no df_comentario e renomear
df_comentario = df_comentario.withColumn("Likes", col("Likes").cast(IntegerType())) \
    .withColumn("Sentiment", col("Sentiment").cast(IntegerType())) \
    .withColumnRenamed("Likes", "Likes Comment")

# 9. Criar campo 'Interaction' no df_video
df_video = df_video.withColumn("Interaction", col("Likes") + col("Comments") + col("Views"))

# 10. Converter 'Published At' para date
df_video = df_video.withColumn("Published At", to_date(col("Published At")))

# 11. Criar campo 'Year' no df_video
df_video = df_video.withColumn("Year", year(col("Published At")))

# 12. Mesclar df_comentario no df_video criando df_join_video_comments
df_join_video_comments = df_video.join(df_comentario, "Video ID", "inner")

# 13. Ler o arquivo 'USvideos.csv' no dataframe 'df_us_videos'
df_us_videos = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("USvideos.csv")

# 14. Mesclar df_us_videos no df_video criando df_join_video_usvideos
df_join_video_usvideos = df_video.join(df_us_videos, "Title", "inner")
df_join_video_usvideos.show()

# 15. Verificar quantidade de campos nulos no df_video
from pyspark.sql.functions import isnan, when, count

df_video.select([count(when(col(c).isNull(), c)).alias(c) for c in df_video.columns]).show()

# 16. Remover coluna '_c0' e salvar df_video como parquet
if '_c0' in df_video.columns:
    df_video = df_video.drop('_c0')
df_video.write \
    .option("header", True) \
    .parquet("videos-tratados-parquet")

# 17. Remover coluna '_c0' e salvar df_join_video_comments como parquet
if '_c0' in df_join_video_comments.columns:
    df_join_video_comments = df_join_video_comments.drop('_c0')
df_join_video_comments.write \
    .option("header", True) \
    .parquet("videos-comments-tratados-parquet")

# Encerrar a sessão Spark
spark.stop()