def print_importance(result, names, top_n):
    order = result.importances_mean.argsort()[::-1][:top_n]
    for i in order:
        print(f"{names[i]}\t{result.importances_mean[i]:.4f}")

print_importance(result=your_bunch, names=X_test.columns, top_n=20)
