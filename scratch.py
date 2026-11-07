from sklearn.metrics import confusion_matrix, precision_recall_curve

TIER = ""
PRECISION_TARGET = 0.90

def get_probs(d):
    words = words_and_chars.transform(d[TEXT_COL])
    return text_model.predict_proba(words)[:, list(text_model.classes_).index(TIER)]

val_probs = get_probs(val)
y_val = (val["tier_notes"] == TIER).astype(int).to_numpy()

precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
cutoff = thresholds[precision[:-1] >= PRECISION_TARGET][0]

for name, d in [("val", val), ("test", test)]:
    y = (d["tier_notes"] == TIER).astype(int).to_numpy()
    pred = (get_probs(d) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(name, "base rate", y.mean(), "precision", tp / (tp + fp), "recall", tp / (tp + fn), "fpr", fp / (fp + tn))
