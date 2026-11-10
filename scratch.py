silver_pd["total_paid"] = silver_pd["total_paid"].fillna(0)
silver_pd["cost_flag"] = np.where(silver_pd["total_paid"] <= 2500, "fast_track", "otherwise")
print(silver_pd["cost_flag"].value_counts())
