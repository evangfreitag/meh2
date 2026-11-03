from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
X_train = hstack([words_and_chars.fit_transform(train["text"]), csr_matrix(normalize(np.vstack(train["embedding"])))])
X_test = hstack([words_and_chars.transform(test["text"]), csr_matrix(normalize(np.vstack(test["embedding"])))])

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, train["tier_notes"])
pred = model.predict(X_test)

print("model on the description vs the notes tier")
print(classification_report(test["tier_notes"], pred))
print(confusion_matrix(test["tier_notes"], pred, labels=TIERS))

# the bar to beat: keywords run on the description alone
print("keywords on the description vs the notes tier")
print(classification_report(test["tier_notes"], test["tier_desc"]))

# eyeball these to sanity check the labels, since there is no SME yet
misses = test.assign(pred=pred)
print(misses[misses["tier_notes"] != misses["pred"]][["text", "tier_notes", "pred"]].head(30).to_string())
