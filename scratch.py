from sklearn.metrics import precision_recall_curve

PRECISION_TARGET = 0.90

y_val = (val[TARGET] == "fast_track").astype(int)
p, r, t = precision_recall_curve(y_val, val_probs)
cutoff = t[p[:-1] >= PRECISION_TARGET][0]
print(cutoff)

test_pred = test_probs >= cutoff



Input: the short loss description at FNOL, plus structured variables from the claim and policy tables
Step 1: TF-IDF on the text into logistic regression, class-balanced
Step 2: that text probability plus the structured variables go into gradient-boosted trees
Output: a Fast Track probability
Labels come from the full claim notes (hindsight). The model only sees what’s known at FNOL
Trained on three labels: keyword rule, cost rule, keyword + cost
Cutoff set on validation, applied unchanged to test
