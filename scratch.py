import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

def fit_flag_model(df, flag_cols, label_col, reg_param):
    data = df.select(*flag_cols, F.col(label_col).cast("double").alias("label")).na.fill(0.0)
    vec = VectorAssembler(inputCols=flag_cols, outputCol="features").transform(data)
    model = LogisticRegression(featuresCol="features", labelCol="label", elasticNetParam=1.0, regParam=reg_param).fit(vec)
    coefs = pd.Series(model.coefficients.toArray(), index=flag_cols)
    return coefs[coefs != 0].sort_values(ascending=False)

display(fit_flag_model(df=claims, flag_cols=flag_columns, label_col="ai_flagged", reg_param=0.01).reset_index())
