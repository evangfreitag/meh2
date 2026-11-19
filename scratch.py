pdf = df.limit(2000).toPandas()
run_viewer(pdf, nlp, matcher, "ai_label", "label", "fast_track", "claim_id", "notes_text")
