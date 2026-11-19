from pyspark.sql import functions as F

def map_keywords(df, match_col, mapping, new_col, sep=","):
    m = F.create_map(*[F.lit(x) for k, v in mapping.items() for x in (k.strip().lower(), v)])
    parts = F.transform(F.split(F.col(match_col), sep), lambda x: F.lower(F.trim(x)))
    mapped = F.filter(F.transform(parts, lambda x: m[x]), lambda x: x.isNotNull())
    return df.withColumn(new_col, F.array_distinct(mapped))

df = map_keywords(df, "match", mapping, "labels")
