def pivot_agg(df, key, col, agg_expr, prefix):
    clean = F.lower(F.trim(F.col(col)))
    clean = F.regexp_replace(clean, r"[^a-z0-9]+", "_")
    clean = F.regexp_replace(clean, r"^_+|_+$", "")
    clean = F.when(clean.isNull() | (clean == ""), F.lit("unknown")).otherwise(clean)
    d = df.withColumn(col, clean)
    out = d.groupBy(key).pivot(col).agg(agg_expr).fillna(0)
    return out.toDF(*[c if c == key else f"{prefix}{c}" for c in out.columns])

flags = pivot_agg(df, "claim_number", "term", F.max("indicator"), "term_")
counts = pivot_agg(df, "claim_number", "term", F.count("term"), "term_")
