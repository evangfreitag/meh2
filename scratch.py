from pyspark.sql import functions as F

MENTION = r"(?i)\b(hospital\w*|admitted|admission|inpatient|discharged|icu)\b"
DATE = r"(\d{1,2}/\d{1,2}/\d{2,4})"


def hospital_mentions(df, text_col):
    return df.filter(F.col(text_col).rlike(MENTION))


def to_date(col):
    four = F.expr(f"try_to_date({col}, 'M/d/yyyy')")
    two = F.expr(f"try_to_date({col}, 'M/d/yy')")
    return F.when(F.col(col).rlike(r"/\d{4}$"), four).otherwise(two)


def extract_hospital(df, text_col):
    t = F.col(text_col)
    admit = F.nullif(F.regexp_extract(t, r"(?i)admitted[^.]{0,40}?" + DATE, 1), F.lit(""))
    disch = F.nullif(F.regexp_extract(t, r"(?i)discharged[^.]{0,40}?" + DATE, 1), F.lit(""))
    d1 = F.nullif(F.regexp_extract(t, r"(?i)(\d+)\s*(?:days?|nights?)\s*(?:in|at)\s*(?:the\s*)?hospital", 1), F.lit(""))
    d2 = F.nullif(F.regexp_extract(t, r"(?i)hospitali[sz]ed\s*(?:for\s*)?(\d+)\s*(?:days?|nights?)", 1), F.lit(""))
    out = (df.withColumn("_admit", admit).withColumn("_disch", disch)
             .withColumn("days_stated", F.coalesce(d1, d2).cast("int")))
    out = (out.withColumn("hospital_admit_date", to_date("_admit"))
              .withColumn("hospital_discharge_date", to_date("_disch"))
              .drop("_admit", "_disch"))
    calc = F.datediff("hospital_discharge_date", "hospital_admit_date")
    return out.withColumn("days_in_hospital", F.coalesce("days_stated", calc))
