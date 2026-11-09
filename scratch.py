import numpy as np

LOSS_COL = "loss_labels"
DISQ_COL = "disq_labels"
COST_COL = "gross_total"
TIER = ""
LOSS_TYPES = {}
DISQUALIFIERS = {}
COST_LIMIT = 2500

def has_any(labels, wanted):
    return hasattr(labels, "__len__") and len(set(labels) & wanted) > 0

df[COST_COL] = df[COST_COL].fillna(0)

text_ok = df[LOSS_COL].apply(has_any, wanted=LOSS_TYPES) & ~df[DISQ_COL].apply(has_any, wanted=DISQUALIFIERS)
cost_ok = df[COST_COL] < COST_LIMIT

df["text_label"] = np.where(text_ok, TIER, "otherwise")
df["cost_label"] = np.where(cost_ok, TIER, "otherwise")
df["both_label"] = np.where(text_ok & cost_ok, TIER, "otherwise")

for col in ["text_label", "cost_label", "both_label"]:
    print(col, df[col].value_counts().to_dict())
