from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

KEYWORD_TABLE = "fnol_keyword_flags"  # output of fnol_keywords.py, the only input
TEXT_COL = "loss_description"
DATE_COL = "loss_date"
TEST_FRACTION = 0.2
TIERS = ["fast_track", "level_2", "level_3", "specialty"]

df = spark.sql(f"""
    SELECT claim_id, tier_notes, tier_desc,
           {TEXT_COL} AS text, {DATE_COL} AS loss_date
    FROM {KEYWORD_TABLE}
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

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)
model.fit(train["text"], train["tier_notes"])
pred = model.predict(test["text"])

print("model on the description vs the notes tier")
print(classification_report(test["tier_notes"], pred))
print(confusion_matrix(test["tier_notes"], pred, labels=TIERS))

# the bar to beat: keywords run on the description alone
print("keywords on the description vs the notes tier")
print(classification_report(test["tier_notes"], test["tier_desc"]))

# eyeball these to sanity check the labels, since there is no SME yet
misses = test.assign(pred=pred)
print(misses[misses["tier_notes"] != misses["pred"]][["text", "tier_notes", "pred"]].head(30).to_string())
