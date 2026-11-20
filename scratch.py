def print_plot_order(ax):
    names = [t.get_text() for t in ax.get_yticklabels()]
    values = [p.get_width() for p in ax.patches]
    for name, value in sorted(zip(names, values), key=lambda x: -x[1]):
        print(f"{name}\t{value:.4f}")

print_plot_order(ax=plt.gca())
