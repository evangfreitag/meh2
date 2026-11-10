POSITIVE = "fast_track"

def get_probs(which):
    if which == "text":
        model, X_val, X_test = text_model, words_val, words_test
    elif which == "structured":
        cols = list(S_train.columns)
        model, X_val, X_test = struct_model, S_val[cols], S_test[cols]
    elif which == "stacked":
        cols = list(C_train.columns)
        X_val = S_val.assign(**{f"text_{t}": text_model.predict_proba(words_val)[:, i] for i, t in enumerate(text_model.classes_)})[cols]
        X_test = S_test.assign(**{f"text_{t}": text_model.predict_proba(words_test)[:, i] for i, t in enumerate(text_model.classes_)})[cols]
        model = stacked_model
    idx = list(model.classes_).index(POSITIVE)
    return model.predict_proba(X_val)[:, idx], model.predict_proba(X_test)[:, idx]

val_probs, test_probs = get_probs("stacked")
