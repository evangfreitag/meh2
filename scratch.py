PRECISION_TARGET = 0.90
VAL = ...
TEST = test

y_val = (VAL["cost_flag"] == "fast_track").astype(int)
y_test = (TEST["cost_flag"] == "fast_track").astype(int)

rows = []
for which in ["text", "structured", "stacked"]:
    val_probs = get_probs(which, VAL)
    test_probs = get_probs(which, TEST)
    p, r, t = precision_recall_curve(y_val, val_probs)
    cutoff = t[p[:-1] >= PRECISION_TARGET][0]
    pred = (test_probs >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "model": which,
        "AP": round(average_precision_score(y_test, test_probs), 3),
        "cutoff": round(cutoff, 4),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "flagged": int(pred.sum()),
        "false_pos": int(fp),
    })

print(pd.DataFrame(rows).to_string(index=False))
