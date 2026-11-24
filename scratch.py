from pyspark.sql import functions as F

def add_length(df, text_col):
    return df.withColumn("n_words", F.size(F.split(F.col(text_col), r"\s+")))

def confidence_by_length(df, confidence_col, bucket_size):
    return (
        df.withColumn("length_bucket", (F.floor(F.col("n_words") / bucket_size) * bucket_size))
        .groupBy("length_bucket")
        .agg(F.count("*").alias("claims"), F.avg(confidence_col).alias("avg_confidence"))
        .orderBy("length_bucket")
    )

negatives = add_length(df=df.where(F.col("severity") == 0), text_col="note_text")
confidence_by_length(df=negatives, confidence_col="confidence", bucket_size=25).show(50)
