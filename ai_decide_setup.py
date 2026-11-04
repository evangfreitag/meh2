import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import OrdinalEncoder

LABEL_COL = "tier_notes"
CAT_COLS = ["state", "policy_type"]  # your categorical columns
NUM_COLS = ["hours_diff"]            # your numeric columns, if any
TIERS = ["fast_track", "level_2", "level_3", "specialty"]

# train and test are the time-split pandas frames from the baseline, with these columns added
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
X_train = pd.concat([pd.DataFrame(enc.fit_transform(train[CAT_COLS].fillna("missing").astype(str)), columns=CAT_COLS, index=train.index), train[NUM_COLS]], axis=1)
X_test = pd.concat([pd.DataFrame(enc.transform(test[CAT_COLS].fillna("missing").astype(str)), columns=CAT_COLS, index=test.index), test[NUM_COLS]], axis=1)

cat_mask = [c in CAT_COLS for c in X_train.columns]
gbt = HistGradientBoostingClassifier(categorical_features=cat_mask, class_weight="balanced", random_state=0)
gbt.fit(X_train, train[LABEL_COL])

print(classification_report(test[LABEL_COL], gbt.predict(X_test)))
proba = gbt.predict_proba(X_test)
print({tier: round(average_precision_score(test[LABEL_COL] == tier, proba[:, i]), 3) for i, tier in enumerate(TIERS)})
