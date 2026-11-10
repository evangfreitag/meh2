Input: the short loss description at FNOL, plus structured variables from the claim and policy tables
Text-only: TF-IDF (words + characters) into logistic regression, class-balanced
Structured-only: the structured variables alone, in gradient-boosted trees
Text + structured: the text model’s out-of-fold probability plus the structured variables, in gradient-boosted trees
Output: a Fast Track probability
Labels come from the full claim notes (hindsight). The models only see what’s known at FNOL
Each model trained on three labels: keyword rule, cost rule, keyword + cost
Cutoff set on validation, applied unchanged to test




from sklearn.metrics import precision_score, recall_score, confusion_matrix

y_test = (test[TARGET] == "fast_track").astype(int)
test_pred = (test_probs >= cutoff).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print("precision", precision_score(y_test, test_pred))
print("recall", recall_score(y_test, test_pred))
print("flagged", test_pred.sum(), "of", len(test_pred))
print("false positives", fp)
