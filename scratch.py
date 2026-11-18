def sample_per_label(df, label_col, key, n, seed=0):
    w1 = Window.partitionBy(label_col, key).orderBy(F.rand(seed))
    one = df.withColumn("_r", F.row_number().over(w1)).filter("_r = 1").drop("_r")
    w2 = Window.partitionBy(label_col).orderBy(F.rand(seed + 1))
    return one.withColumn("_rn", F.row_number().over(w2)).filter(F.col("_rn") <= n).drop("_rn")
