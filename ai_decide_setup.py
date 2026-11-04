from pyspark.sql import functions as F

LATE_COL = "date_a"   # flagged when this date is later than the other one
EARLY_COL = "date_b"

df = (df
      .withColumn(LATE_COL, F.to_date(LATE_COL))
      .withColumn(EARLY_COL, F.to_date(EARLY_COL))
      .withColumn("days_diff", F.datediff(LATE_COL, EARLY_COL))
      .withColumn("date_fraud_flag", (F.col("days_diff") > 0).cast("int")))
