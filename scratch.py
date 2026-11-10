def label_counts(frames, label_col):
    rows = []
    for name, d in frames.items():
        n = len(d)
        fast = (d[label_col] == "fast_track").sum()
        rows.append({
            "set": name,
            "claims": f"{n:,}",
            "fast_track": f"{fast:,} ({fast / n:.1%})",
            "otherwise": f"{n - fast:,} ({(n - fast) / n:.1%})",
        })
    return pd.DataFrame(rows)

display(label_counts({"train": train, "validation": VAL, "test": test, "silver": silver_pd}, "cost_flag"))
