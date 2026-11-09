CAT_COLS = ["term", "vehicle_type", "road_type"]

result = df.select(ID_COL).distinct()

for c in CAT_COLS:
    clean = F.lower(F.trim(F.regexp_replace(F.regexp_replace(F.col(c), "/", "_"), r"[^\w\s]", "")))
    counts = df.withColumn(c, clean).groupBy(ID_COL).pivot(c).count().fillna(0)
    counts = counts.toDF(ID_COL, *[f"{c}_{x}" for x in counts.columns if x != ID_COL])
    result = result.join(counts, on=ID_COL, how="left")
