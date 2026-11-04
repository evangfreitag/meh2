from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

KEYWORD_TABLE = "fnol_keyword_flags"  # output of fnol_keywords.py, the only input
TEXT_COL = "loss_description"
DATE_COL = "loss_date"
TEST_FRACTION = 0.2
EMBED_TABLE = "fnol_embeddings"  # output of fnol_embed.py
TIERS = ["fast_track", "level_2", "level_3", "specialty"]
HIDDEN = 256  # size of the MLP hidden layer
C_VALUES = [0.1, 1, 10]  # regularization strengths to try on the embeddings

df = spark.sql(f"""
    SELECT k.claim_id, k.tier_notes, k.tier_desc,
           k.{TEXT_COL} AS text, k.{DATE_COL} AS loss_date, e.embedding
    FROM {KEYWORD_TABLE} k
    JOIN {EMBED_TABLE} e ON k.claim_id = e.claim_id
""").toPandas()

# no real label when even the full notes had no evidence
df = df.dropna(subset=["text"])
df = df[df["tier_notes"] != "unknown"]

# time split: oldest claims train, newest claims test
df = df.sort_values("loss_date")
cut = int(len(df) * (1 - TEST_FRACTION))
train, test = df.iloc[:cut], df.iloc[cut:]
print(train["tier_notes"].value_counts())
print(test["tier_notes"].value_counts())

# words and character pieces (catches typos and variants), then the embeddings next to them
# embeddings are scaled to unit length because the GTE model does not do that itself
words_and_chars = FeatureUnion([
    ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
])
words_train = words_and_chars.fit_transform(train["text"])
words_test = words_and_chars.transform(test["text"])
emb_train = normalize(np.vstack(train["embedding"]))
emb_test = normalize(np.vstack(test["embedding"]))
X_train = hstack([words_train, csr_matrix(emb_train)])
X_test = hstack([words_test, csr_matrix(emb_test)])

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, train["tier_notes"])
pred = model.predict(X_test)

# same model on the words alone, so the comparison is fair
text_only = LogisticRegression(class_weight="balanced", max_iter=1000)
text_only.fit(words_train, train["tier_notes"])

print("words only vs the notes tier")
print(classification_report(test["tier_notes"], text_only.predict(words_test)))

print("words + embeddings vs the notes tier")
print(classification_report(test["tier_notes"], pred))
print(confusion_matrix(test["tier_notes"], pred, labels=TIERS))

# embeddings alone, with a few regularization strengths (default C=1 may be too tight for dense features)
emb_models = []
for C in C_VALUES:
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    m.fit(emb_train, train["tier_notes"])
    emb_models.append((f"embeddings only, C={C}", m, emb_test))

# small neural net on the embeddings; sklearn has no class_weight, so divide by the class priors after
mlp = MLPClassifier(hidden_layer_sizes=(HIDDEN,), early_stopping=True, random_state=0)
mlp.fit(emb_train, train["tier_notes"])
prior = train["tier_notes"].value_counts(normalize=True)[mlp.classes_].to_numpy()
mlp_pred = mlp.classes_[(mlp.predict_proba(emb_test) / prior).argmax(axis=1)]
print("MLP on embeddings vs the notes tier")
print(classification_report(test["tier_notes"], mlp_pred))

# average precision ignores the cutoff, so it shows which features rank claims better
print("average precision per tier, higher is better")
for name, m, X in [("words only", text_only, words_test), ("words + embeddings", model, X_test)] + emb_models + [(f"MLP {HIDDEN}", mlp, emb_test)]:
    proba = m.predict_proba(X)
    print(name, {str(tier): round(average_precision_score(test["tier_notes"] == tier, proba[:, i]), 3) for i, tier in enumerate(m.classes_)})

# the bar to beat: keywords run on the description alone
print("keywords on the description vs the notes tier")
print(classification_report(test["tier_notes"], test["tier_desc"]))

# eyeball these to sanity check the labels, since there is no SME yet
misses = test.assign(pred=pred)
print(misses[misses["tier_notes"] != misses["pred"]][["text", "tier_notes", "pred"]].head(30).to_string())
