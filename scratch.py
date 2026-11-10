for d in (train, val, test):
    d["cost_flag"] = np.where(d["gross_total"].fillna(0) <= 2500, "fast_track", "otherwise")
    print(d["cost_flag"].value_counts().to_dict())
