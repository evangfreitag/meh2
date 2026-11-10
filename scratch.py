COST_COL = "total_paid"

def fp_cost(data, probs, cutoff, label_col, cost_col):
    fp = data[(probs >= cutoff) & (data[label_col] != "fast_track")]
    return {
        "false_pos": len(fp),
        "fp_total_cost": round(fp[cost_col].sum()),
        "fp_median_cost": round(fp[cost_col].median()),
        "fp_mean_cost": round(fp[cost_col].mean()),
    }
