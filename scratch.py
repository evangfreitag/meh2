import pandas as pd

def as_list(v):
    if isinstance(v, str):
        return [v]
    if v is None or isinstance(v, float):
        return ["none"]
    return [str(c) for c in v] or ["none"]

def confusion_table(df, pred_col, truth_col):
    pairs = pd.DataFrame({"truth": df[truth_col].apply(as_list), "pred": df[pred_col].apply(as_list)})
    pairs = pairs.explode("truth").reset_index(drop=True).explode("pred")
    return pd.crosstab(pairs["truth"], pairs["pred"])

def class_metrics(df, pred_col, truth_col):
    rows = []
    for cls in sorted(set(all_classes(df, pred_col)) | set(all_classes(df, truth_col))):
        m = outcome_mask(df, pred_col, truth_col, cls)
        tp, fp, fn, tn = (int(m[k].sum()) for k in ("tp", "fp", "fn", "tn"))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        rows.append({"class": cls, "truth_positives": tp + fn, "predicted_positives": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": None if precision is None else round(precision, 3),
                     "recall": None if recall is None else round(recall, 3),
                     "f1": None if f1 is None else round(f1, 3)})
    return pd.DataFrame(rows)

display(confusion_table(df, "spacy_label", "ai_label"))
display(class_metrics(df, "spacy_label", "ai_label"))
