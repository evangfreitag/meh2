import matplotlib.pyplot as plt
def plot_kde(df, col, group_col):
    pdf = df.select(col, group_col).toPandas()
    pdf.groupby(group_col)[col].plot.kde(legend=True)
    plt.xlabel(col)
    plt.xlim(0, 1)
    plt.show()

plot_kde(df=classified, col="predicted_probability", group_col="ai_flagged")
