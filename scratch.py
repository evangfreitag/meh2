import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

def add_bin(df, col, bin_width, new_col):
    return df.withColumn(new_col, F.floor(F.round(F.col(col) / bin_width, 6)) * bin_width)

def rate_by_bin(df, bin_col, ai_col):
    return (
        df.groupBy(bin_col)
        .agg(F.count("*").alias("claims"), F.avg(F.col(ai_col).cast("double")).alias("ai_rate"))
        .orderBy(bin_col)
    )

def plot_rate_by_bin(rates, bin_col):
    pdf = rates.toPandas()
    fig, ax = plt.subplots()
    ax.plot(pdf[bin_col], pdf["ai_rate"], marker="o")
    ax.set_xlabel(bin_col)
    ax.set_ylabel("share flagged by AI")
    plt.show()

def auc(df, prob_col, label_col):
    data = (
        df.where(F.col(label_col).isNotNull() & F.col(prob_col).isNotNull())
        .select(F.col(prob_col).cast("double").alias("prob"), F.col(label_col).cast("double").alias("label"))
    )
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prob", labelCol="label", metricName="areaUnderROC")
    return evaluator.evaluate(data)

# example calls
binned = add_bin(df=df, col="predicted_probability", bin_width=0.05, new_col="prob_bin")
rates = rate_by_bin(df=binned, bin_col="prob_bin", ai_col="ai_flagged")
display(rates)
plot_rate_by_bin(rates=rates, bin_col="prob_bin")
print(auc(df=df, prob_col="predicted_probability", label_col="ai_flagged"))
