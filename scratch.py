import re
import html
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output

def highlight_html(text, spans):
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f'<mark style="background:#ffe08a">{html.escape(text[s:e])}</mark>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)

def get_spans(keywords, text):
    kws = sorted({k.strip() for k in (keywords or "").split(",") if k.strip()}, key=len, reverse=True)
    if not kws:
        return []
    pat = re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]

def has_class(value, cls):
    if isinstance(value, str):
        return value == cls
    if value is None or isinstance(value, float):      # null (None or NaN)
        return False
    return cls in list(value)

def outcome_mask(df, pred_col, truth_col, positive):
    pred = df[pred_col].apply(lambda v: has_class(v, positive))
    actual = df[truth_col].apply(lambda v: has_class(v, positive))
    return {"fp": pred & ~actual, "tp": pred & actual, "fn": ~pred & actual, "tn": ~pred & ~actual}

def all_classes(df, col):
    found = set()
    for v in df[col]:
        if v is None or (isinstance(v, float) and v != v):
            continue
        found |= {v} if isinstance(v, str) else {str(c) for c in v}
    return sorted(found)

def run_viewer(df, pred_col, truth_col, id_col, text_col, kw_col):
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
            display(HTML(f"<div style='max-width:900px;line-height:1.5'>{head}<hr>{highlight_html(text, get_spans(kws, text))}</div>"))

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
