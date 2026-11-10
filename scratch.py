print(len(test), test["total_paid"].isna().sum(), (test["total_paid"] <= 2500).sum(), test["cost_flag"].eq("fast_track").sum())
