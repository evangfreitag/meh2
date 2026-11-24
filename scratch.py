from pyspark.sql import functions as F

def exploded_flags(df, prob_col, flags_col, scores_col):
    pairs = F.arrays_zip(F.col(flags_col).alias("flag"), F.col(scores_col).alias("score"))
    return (
        df.withColumn("pair", F.explode(pairs))
        .withColumn("flag", F.col("pair.flag"))
        .withColumn("score", F.col("pair.score"))
        .drop("pair", flags_col, scores_col)
        .orderBy(F.desc(prob_col), F.desc("score"))
    )

exploded_flags(df=df, prob_col="predicted_probability", flags_col="flags", scores_col="scores").show(50, False)
