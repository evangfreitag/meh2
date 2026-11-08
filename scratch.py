from pyspark.sql import functions as F

total = df.count()

nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).toPandas().T
nulls.columns = ["nulls"]
nulls["null_share"] = (nulls["nulls"] / total).round(3)
nulls["non_null"] = total - nulls["nulls"]

print(nulls.sort_values("null_share", ascending=False).to_string())
