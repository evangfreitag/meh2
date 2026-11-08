CUTOFF = 0.7
TIER = ""
COST_COL = ""
TARGET = ""

test = test.assign(prob=test_probs)
test["flagged"] = test["prob"] >= CUTOFF
test["is_true"] = test[TARGET] == TIER

fp = test[test["flagged"] & ~test["is_true"]]
tp = test[test["flagged"] & test["is_true"]]

print("false positives", len(fp), "total", fp[COST_COL].sum(), "median", fp[COST_COL].median(), "mean", fp[COST_COL].mean())
print("true positives ", len(tp), "total", tp[COST_COL].sum(), "median", tp[COST_COL].median(), "mean", tp[COST_COL].mean())
print(fp.groupby(TARGET)[COST_COL].agg(["count", "median", "sum"]))





CUTOFF = 0.7
TIER = ""
COST_COL = ""
TARGET = ""

flagged = test[test_probs >= CUTOFF]
fp = flagged[flagged[TARGET] != TIER]

print("false positives", len(fp), COST_COL, fp[COST_COL].sum())
print("share of flagged cost", fp[COST_COL].sum() / flagged[COST_COL].sum())

for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9]:
    flagged = test[test_probs >= cutoff]
    fp = flagged[flagged[TARGET] != TIER]
    print(cutoff, "flagged", len(flagged), "fp", len(fp), "fp gross", round(fp[COST_COL].sum()), "fp median", round(fp[COST_COL].median()))
