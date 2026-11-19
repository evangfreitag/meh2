from pyspark.sql import functions as F, Window

def build_run_dates(df, run_id_col, date_col, out_col="run_date"):
    return df.groupBy(run_id_col).agg(F.max(date_col).alias(out_col))

def notes_since_last_run(runs, run_dates, notes, key, run_id_col, run_date_col, note_date_col, note_col, sep=" "):
    r = runs.join(run_dates, run_id_col)
    w = Window.partitionBy(key).orderBy(run_date_col, run_id_col)
    r = r.withColumn("_prev", F.lag(run_date_col).over(w))
    n = notes.select(F.col(key).alias("_k"), note_date_col, note_col)
    cond = ((r[key] == n["_k"]) & (n[note_date_col] <= r[run_date_col])
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

def notes_for_runs(model_output, notes, key, run_id_col, score_date_col, note_date_col, note_col, sep=" "):
    run_dates = build_run_dates(model_output, run_id_col, score_date_col)
    runs = model_output.select(key, run_id_col).distinct()
    return notes_since_last_run(runs, run_dates, notes, key, run_id_col, "run_date", note_date_col, note_col, sep)

def latest_rows(df, key, seq_col):
    w = Window.partitionBy(key).orderBy(F.col(seq_col).desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

out = notes_for_runs(model_output, notes, "claim_id", "model_run_id", "score_date", "note_date", "note_text")
latest = latest_rows(out, "claim_id", "model_run_id")
