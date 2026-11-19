import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve


def synthetic_features(X, y, level, rng):
    y = np.asarray(y)
    n = len(y)
    flag = rng.random(n) < np.where(y == 1, 0.3 + 0.4 * level, 0.3)
    base = np.full(5, 0.2)
    skew = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
    probs = np.where(y[:, None] == 1, (1 - level) * base + level * skew, base)
    cat = (probs.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1)
    score = 2 * level * (2 * y - 1) + rng.normal(size=n)
    return X.assign(synthetic_injury_flag=flag.astype(int),
                    synthetic_loss_type=pd.Categorical(cat, categories=range(5)),
                    synthetic_severity_score=score)


def coverage_at_precision(p_val, y_val, p_test, target):
    p, r, t = precision_recall_curve(y_val, p_val)
    ok = t[p[:-1] >= target]
    return 0.0 if len(ok) == 0 else float((p_test >= ok[0]).mean())


def sensitivity(Xtr, ytr, Xva, yva, Xte, yte, levels, seeds, target):
    rows = []
    for lv in levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            A, B, C = [synthetic_features(X, y, lv, rng) for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte))]
            m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed).fit(A, ytr)
            pva, pte = m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]
            rows.append((lv, average_precision_score(yte, pte), coverage_at_precision(pva, yva, pte, target)))
    r = pd.DataFrame(rows, columns=["level", "avg_precision", "fast_track_rate"])
    return r.groupby("level").agg(["mean", "std"]).round(3)
