import re
import html
from IPython.display import HTML, display
from ipywidgets import interact, IntSlider

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

def outcome_mask(pdf, pred_col, label_col, positive):
    pred = pdf[pred_col].apply(lambda a: a is not None and positive in list(a))   # AI label is an array
    actual = pdf[label_col] == positive                                           # other label is a string
    return {"fp": pred & ~actual, "tp": pred & actual, "fn": ~pred & actual, "tn": ~pred & ~actual}

def all_classes(pdf, pred_col, label_col):
    ai = {str(c) for a in pdf[pred_col] if a is not None for c in a}
    return sorted(ai | set(pdf[label_col].dropna()))

def make_viewer(pdf, pred_col, label_col, id_col, text_col, kw_col):
    def show(cls, which="fp", i=0):
        sub = pdf[outcome_mask(pdf, pred_col, label_col, cls)[which]]
        if sub.empty:
            return display(HTML(f"no {which} rows for {html.escape(str(cls))}"))
        r = sub.iloc[min(i, len(sub) - 1)]
        head = f"<b>{html.escape(str(r[id_col]))}</b> &nbsp; {pred_col}={html.escape(str(r[pred_col]))} &nbsp; {label_col}={html.escape(str(r[label_col]))} &nbsp; ({min(i, len(sub)-1) + 1} of {len(sub)})"
        body = highlight_html(r[text_col], get_spans(r[kw_col], r[text_col]))
        display(HTML(f"<div style='max-width:900px;line-height:1.5'>{head}<hr>{body}</div>"))
    return show

def run_viewer(pdf, pred_col, label_col, id_col, text_col, kw_col):
    show = make_viewer(pdf, pred_col, label_col, id_col, text_col, kw_col)
    interact(show, cls=all_classes(pdf, pred_col, label_col), which=["fp", "tp", "fn", "tn"],
             i=IntSlider(min=0, max=len(pdf) - 1, step=1, value=0))

pdf = df.limit(2000).toPandas()
run_viewer(pdf, "ai_label", "spacy_label", "claim_id", "notes_text", "keywords")
