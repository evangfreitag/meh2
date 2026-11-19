import re

def get_spans(keywords, text):
    kws = sorted({k for k in (keywords if keywords is not None else []) if k}, key=len, reverse=True)
    if not kws:
        return []
    pat = re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE)
    return [(m.start(), m.end(), "") for m in pat.finditer(text)]

pdf = df.limit(2000).toPandas()
run_viewer(pdf, nlp, matcher, "ai_label", "label", "fast_track", "claim_id", "notes_text")
