from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd

ID_COL = "claim_id"
LABEL_COL = "tier_notes"
CAT_COLS = ["state", "policy_type"]   # your categorical columns
NUM_COLS = ["hours_diff"]             # your numeric columns
TEST_FRACTION = 0.2


# no real label when even the full notes had no evidence
df = df.dropna(subset=[TEXT_COL])
df = df[df[LABEL_COL] != "unknown"]

# time split: oldest claims train, newest claims test
df = df.sort_values(DATE_COL)
cut = int(len(df) * (1 - TEST_FRACTION))
train, test = df.iloc[:cut], df.iloc[cut:]
print(train[LABEL_COL].value_counts())
print(test[LABEL_COL].value_counts())

# text model: words and character pieces into logistic regression
words_and_chars = FeatureUnion([
    ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
])
words_train = words_and_chars.fit_transform(train[TEXT_COL])
words_test = words_and_chars.transform(test[TEXT_COL])
text_model = LogisticRegression(class_weight="balanced", max_iter=1000)
text_model.fit(words_train, train[LABEL_COL])

# structured columns: categories become number codes, unseen categories become missing
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
def struct_features(d, fit=False):
    cats = d[CAT_COLS].fillna("missing").astype(str)
    codes = enc.fit_transform(cats) if fit else enc.transform(cats)
    return pd.concat([pd.DataFrame(codes, columns=CAT_COLS, index=d.index), d[NUM_COLS]], axis=1)
S_train = struct_features(train, fit=True)
S_test = struct_features(test)

# text probabilities as features; train ones are out-of-fold so the trees don't see a model that already saw the answer
oof = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=1000), words_train, train[LABEL_COL], cv=5, method="predict_proba")
C_train = S_train.assign(**{f"text_{t}": oof[:, i] for i, t in enumerate(TIERS)})
C_test = S_test.assign(**{f"text_{t}": text_model.predict_proba(words_test)[:, i] for i, t in enumerate(TIERS)})

def fit_trees(X):
    cat_mask = [c in CAT_COLS for c in X.columns]
    return HistGradientBoostingClassifier(categorical_features=cat_mask, class_weight="balanced", random_state=0).fit(X, train[LABEL_COL])
struct_model = fit_trees(S_train)
combined_model = fit_trees(C_train)

print("text only")
print(classification_report(test[LABEL_COL], text_model.predict(words_test)))
print("structured only")
print(classification_report(test[LABEL_COL], struct_model.predict(S_test)))
print("text + structured")
print(classification_report(test[LABEL_COL], combined_model.predict(C_test)))

# average precision ignores the cutoff, so it compares how well each one ranks claims
print("average precision per tier, higher is better")
for name, m, X in [("text only", text_model, words_test), ("structured only", struct_model, S_test), ("text + structured", combined_model, C_test)]:
    proba = m.predict_proba(X)
    print(name, {tier: round(average_precision_score(test[LABEL_COL] == tier, proba[:, i]), 3) for i, tier in enumerate(TIERS)})
