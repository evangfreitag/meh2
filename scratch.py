from pyspark.sql import Window, functions as F

obj = [c for c in num_cols if X[c].dtype == "object"]
X[obj] = X[obj].apply(pd.to_numeric, errors="coerce")


def within_days(df, col_a, col_b, new_col, threshold, flag_col):
    gap = F.abs(F.datediff(F.to_date(col_a), F.to_date(col_b)))
    ok = (gap <= threshold) & F.col(flag_col).cast("boolean")
    return df.withColumn(new_col, F.coalesce(ok.cast("int"), F.lit(0)))


def bucket_rare(df, col, min_count, other="other"):
    v = F.when(F.trim(F.col(col)) == "", None).otherwise(F.col(col))
    n = F.count(F.lit(1)).over(Window.partitionBy(v))
    return df.withColumn(col, F.when(v.isNull(), None).when(n < min_count, F.lit(other)).otherwise(v))
    
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, BooleanType


def occurs(df, c):
    t = df.schema[c].dataType
    if isinstance(t, NumericType):
        return F.col(c).isNotNull() & (F.col(c) != 0)
    if isinstance(t, BooleanType):
        return F.coalesce(F.col(c), F.lit(False))
    return F.col(c).isNotNull()

def drop_sparse(df, min_count):
    counts = df.select([F.count(F.when(occurs(df, c), 1)).alias(c) for c in df.columns]).first().asDict()
    return df.drop(*[c for c in df.columns if counts[c] < min_count])

def within_days(df, col_a, col_b, new_col, threshold):
    gap = F.abs(F.datediff(F.to_date(col_a), F.to_date(col_b)))
    return df.withColumn(new_col, F.coalesce((gap <= threshold).cast("int"), F.lit(0)))

AMBULANCE = r"(?i)\b(ambulance|paramedics?)\b|\bEMS\b"
ER = r"(?i)\b(emergency (room|department|dept)|urgent care)\b|\bER\b"

import numpy as np


def add_flag(df, label_col, name, rate, lift, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: (rng.random(len(y)) < np.where(y == 1, rate + lift, rate)).astype(int)})


def add_category(df, label_col, name, pos_probs, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    k = len(pos_probs)
    probs = np.where(y[:, None] == 1, np.array(pos_probs), np.full(k, 1 / k))
    cat = (probs.cumsum(axis=1) < rng.random(len(y))[:, None]).sum(axis=1)
    return df.assign(**{name: cat})


def add_score(df, label_col, name, gap, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: gap * (2 * y - 1) / 2 + rng.normal(size=len(y))})

def add_count(df, label_col, name, rate, lift, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: rng.poisson(np.where(y == 1, rate + lift, rate))})


data_syn = (data
    .pipe(add_flag, "notes", "synthetic_i", 0.3, 0.4, 1)
    .pipe(add_category, "notes", "synthetic_lt", [0.05, 0.10, 0.20, 0.30, 0.35], 3)
    .pipe(add_score, "notes", "synthetic_s", 4, 4)
    .pipe(add_count, "notes", "synthetic_ip", 1.0, 0.8, 6))




# Spark v
import math
from pyspark.sql import functions as F


def add_flag(df, label_col, name, rate, lift, seed):
    p = F.when(F.col(label_col) == 1, rate + lift).otherwise(rate)
    return df.withColumn(name, (F.rand(seed) < p).cast("int"))


def add_category(df, label_col, name, pos_probs, seed):
    k = len(pos_probs)
    pos_cum = [sum(pos_probs[:i + 1]) for i in range(k - 1)]
    neg_cum = [(i + 1) / k for i in range(k - 1)]
    u = F.rand(seed)
    hits = [F.when(u > F.when(F.col(label_col) == 1, pc).otherwise(nc), 1).otherwise(0)
            for pc, nc in zip(pos_cum, neg_cum)]
    return df.withColumn(name, sum(hits))


def add_score(df, label_col, name, gap, seed):
    return df.withColumn(name, gap * (2 * F.col(label_col) - 1) / 2 + F.randn(seed))


def poisson_cdf(lam, upto):
    return [sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(i + 1)) for i in range(upto)]


def add_count(df, label_col, name, rate, lift, seed):
    upto = int(max(rate, rate + lift) * 4 + 10)
    pos, neg = poisson_cdf(rate + lift, upto), poisson_cdf(rate, upto)
    u = F.rand(seed)
    hits = [F.when(u > F.when(F.col(label_col) == 1, p).otherwise(n), 1).otherwise(0)
            for p, n in zip(pos, neg)]
    return df.withColumn(name, sum(hits))


data_syn = (data
    .transform(add_flag, "notes", "synthetic_i", 0.3, 0.4, 1)
    .transform(add_category, "notes", "synthetic_lt", [0.05, 0.10, 0.20, 0.30, 0.35], 3)
    .transform(add_score, "notes", "synthetic_s", 4, 4)
    .transform(add_count, "notes", "synthetic_ip", 1.0, 0.8, 6))
data_syn = data_syn.cache()
