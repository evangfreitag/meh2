import numpy as np


def add_synthetic(df, label_col, flag_rate, flag_lift, loss_probs, score_gap, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    n = len(df)
    k = len(loss_probs)
    flag = rng.random(n) < np.where(y == 1, flag_rate + flag_lift, flag_rate)
    probs = np.where(y[:, None] == 1, np.array(loss_probs), np.full(k, 1 / k))
    cat = (probs.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1)
    return df.assign(synthetic_flag=flag.astype(int),
                     synthetic_type=cat,
                     synthetic_score=score_gap * (2 * y - 1) / 2 + rng.normal(size=n))


strong = add_synthetic(data, "tier_notes", 0.3, 0.4, [0.05, 0.10, 0.20, 0.30, 0.35], 4, 0)
medium = add_synthetic(data, "tier_notes", 0.3, 0.2, [0.12, 0.16, 0.20, 0.24, 0.28], 2, 0)
weak   = add_synthetic(data, "tier_notes", 0.3, 0.1, [0.16, 0.18, 0.20, 0.22, 0.24], 1, 0)
none   = add_synthetic(data, "tier_notes", 0.3, 0.0, [0.2, 0.2, 0.2, 0.2, 0.2], 0, 0)
