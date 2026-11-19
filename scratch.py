def ai_detail_html(labels, scores, rationales, selected):
    def seq(v):
        return [] if v is None or isinstance(v, float) else list(v)
    labels, scores, rationales = seq(labels), seq(scores), seq(rationales)
    rows = []
    for i in range(max(len(labels), len(scores), len(rationales))):
        label = labels[i] if i < len(labels) else "?"
        score = scores[i] if i < len(scores) else None
        reason = rationales[i] if i < len(rationales) else None
        score_txt = "" if score is None or score != score else f" &nbsp; score {float(score):.2f}"
        mark = " &nbsp; <b>[selected class]</b>" if label == selected else ""
        rows.append(f"<div style='margin:6px 0'><b>{html.escape(str(label))}</b>{score_txt}{mark}<br>{html.escape(str(reason or ''))}</div>")
    return "".join(rows) or "AI: no labels"

def run_viewer(df, pred_col, truth_col, id_col, text_col, kw_col, score_col, rationale_col):
    cls_dd = widgets.Dropdown(options=all_classes(df, pred_col), description="class")
    which_dd = widgets.Dropdown(options=[("false positive", "fp"), ("true positive", "tp"), ("false negative", "fn"), ("true negative", "tn")], description="outcome")
    claim_dd = widgets.Dropdown(description="claim")
    out = widgets.Output()

    def group():
        return df[outcome_mask(df, pred_col, truth_col, cls_dd.value)[which_dd.value]]

    def render(_=None):
        sub = group()
        with out:
            clear_output(wait=True)
            if sub.empty or claim_dd.value is None:
                display(HTML("no claims in this group"))
                return
            r = sub.iloc[claim_dd.value]
            text, kws = r[text_col], r[kw_col]
            n = len(get_spans(kws, text))
            head = (f"<b>{html.escape(str(r[id_col]))}</b> &nbsp; {pred_col}={html.escape(str(r[pred_col]))} &nbsp; {truth_col}={html.escape(str(r[truth_col]))}"
                    f"<br>keywords: {html.escape(str(kws))} &nbsp; ({n} highlighted) &nbsp; text length: {len(text)}")
            ai = ai_detail_html(r[truth_col], r[score_col], r[rationale_col], cls_dd.value)
            display(HTML(f"<div style='max-width:900px;line-height:1.5'>{head}<hr>{ai}<hr>{highlight_html(text, get_spans(kws, text))}</div>"))

    def refresh(_=None):
        sub = group()
        claim_dd.unobserve(render, names="value")
        claim_dd.options = [(str(v), i) for i, v in enumerate(sub[id_col])]
        claim_dd.value = 0 if len(sub) else None
        claim_dd.observe(render, names="value")
        render()

    cls_dd.observe(refresh, names="value")
    which_dd.observe(refresh, names="value")
    claim_dd.observe(render, names="value")
    display(widgets.HBox([cls_dd, which_dd, claim_dd]), out)
    refresh()
