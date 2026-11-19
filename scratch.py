from IPython.display import HTML, display
from ipywidgets import interact, IntSlider

def outcome_mask(pdf, pred_col, label_col, positive):
    pred, actual = pdf[pred_col] == positive, pdf[label_col] == positive
    return {"fp": pred & ~actual, "tp": pred & actual, "fn": ~pred & actual, "tn": ~pred & ~actual}

def make_viewer(pdf, nlp, matcher, pred_col, label_col, positive, id_col, text_col):
    masks = outcome_mask(pdf, pred_col, label_col, positive)
    def show(which="fp", i=0):
        sub = pdf[masks[which]]
        if sub.empty:
            return display(HTML(f"no {which} rows"))
        r = sub.iloc[min(i, len(sub) - 1)]
        head = f"<b>{html.escape(str(r[id_col]))}</b> &nbsp; {pred_col}={html.escape(str(r[pred_col]))} &nbsp; {label_col}={html.escape(str(r[label_col]))} &nbsp; ({min(i, len(sub)-1) + 1} of {len(sub)})"
        body = highlight_html(r[text_col], get_spans(nlp, matcher, r[text_col]))
        display(HTML(f"<div style='max-width:900px;line-height:1.5'>{head}<hr>{body}</div>"))
    return show

def run_viewer(pdf, nlp, matcher, pred_col, label_col, positive, id_col, text_col):
    show = make_viewer(pdf, nlp, matcher, pred_col, label_col, positive, id_col, text_col)
    interact(show, which=["fp", "tp", "fn", "tn"], i=IntSlider(min=0, max=len(pdf) - 1, step=1, value=0))
