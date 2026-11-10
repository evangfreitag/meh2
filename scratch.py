import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, confusion_matrix



def get_probs(which, data):
    words = words_and_chars.transform(data[TEXT_COL])
    S = struct_features(data)[list(S_train.columns)]
    if which == "text":
        model, X = text_model, words
    elif which == "structured":
        model, X = struct_model, S
    else:
        X = S.assign(**{f"text_{t}": text_model.predict_proba(words)[:, i] for i, t in enumerate(text_model.classes_)})[list(C_train.columns)]
        model = stacked_model
    return model.predict_proba(X)[:, list(model.classes_).index("fast_track")]

probs = get_probs("stacked", EVAL)




PRECISION_TARGET = 0.90
y_val = (val["cost_flag"] == "fast_track").astype(int)
y_test = (test["cost_flag"] == "fast_track").astype(int)

rows = []
for which in ["text", "structured", "stacked"]:
    val_probs, test_probs = get_probs(which)
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
