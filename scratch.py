from sklearn.metrics import precision_recall_curve, precision_score, recall_score

col = list(text_model.classes_).index("fast_track")
proba = text_model.predict_proba(words_test)[:, col]
y_true = (test[LABEL_COL] == "fast_track").to_numpy()
y_default = text_model.predict(words_test) == "fast_track"

print("classes", text_model.classes_)
print("positives", y_true.sum(), "of", len(y_true))
print("default cutoff: precision", precision_score(y_true, y_default), "recall", recall_score(y_true, y_default))

precision, recall, cutoff = precision_recall_curve(y_true, proba)
print("best precision on the curve", precision[:-1].max())
