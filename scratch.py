PRED_COL = "pred"
COST_COL = "incurred"
EXPENSIVE_THRESHOLD = 1

fast_tracked = df[df[PRED_COL] == POSITIVE_LABEL]
wrong = fast_tracked[fast_tracked[NEW_COL] != POSITIVE_LABEL]

# share of fast-tracked dollars that were wrongly fast tracked
dollar_leakage = wrong[COST_COL].sum() / fast_tracked[COST_COL].sum()

# share of expensive claims that got fast tracked
expensive = df[df[COST_COL] >= EXPENSIVE_THRESHOLD]
expensive_fast_tracked_rate = (expensive[PRED_COL] == POSITIVE_LABEL).mean()
