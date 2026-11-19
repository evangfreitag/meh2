from pyspark.sql import functions as F, Window

def notes_since_last_run(runs, run_dates, notes, key, run_id_col, run_date_col, note_date_col, note_col, window_days, sep=" "):
    r = runs.join(run_dates, run_id_col)
    w = Window.partitionBy(key).orderBy(run_date_col, run_id_col)
    r = r.withColumn("_prev", F.lag(run_date_col).over(w))
    n = notes.select(F.col(key).alias("_k"), note_date_col, note_col)
    cond = ((r[key] == n["_k"]) & (n[note_date_col] <= r[run_date_col])
            & (n[note_date_col] > F.date_sub(r[run_date_col], window_days))
            & (r["_prev"].isNull() | (n[note_date_col] > r["_prev"])))
    j = r.join(n, cond, "left")
    has_note = F.col(note_col).isNotNull()
    agg = (j.groupBy(key, run_id_col)
            .agg(F.sort_array(F.collect_list(F.when(has_note, F.struct(note_date_col, note_col)))).alias("_s"),
                 F.count(F.when(has_note, 1)).alias("n_notes"),
                 F.min(F.when(has_note, F.col(note_date_col))).alias("first_note"),
                 F.max(F.when(has_note, F.col(note_date_col))).alias("last_note")))
    text = F.concat_ws(sep, F.col(f"_s.{note_col}"))
    return (agg.withColumn("notes_text", F.when(F.size("_s") > 0, text)).drop("_s")
               .join(r.drop("_prev"), [key, run_id_col]))

out = notes_since_last_run(runs, run_dates, notes, "claim_id", "model_run_id", "run_date", "note_date", "note_text", 7)
out.select(F.max(F.datediff("last_note", "first_note")).alias("widest_span_days")).show()   # should be 7 or less
