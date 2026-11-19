import numpy as np


def add_flag(df, label_col, name, rate, lift, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: (rng.random(len(y)) < np.where(y == 1, rate + lift, rate)).astype(int)})


def add_category(df, label_col, name, pos_probs, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    k = len(pos_probs)
    probs = np.where(y[:, None] == 1, np.array(pos_probs), np.full(k, 1 / k))
    cat = (probs.cumsum(axis=1) < rng.random(len(y))[:, None]).sum(axis=1)
    return df.assign(**{name: cat})


def add_score(df, label_col, name, gap, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: gap * (2 * y - 1) / 2 + rng.normal(size=len(y))})

def add_count(df, label_col, name, rate, lift, seed):
    rng = np.random.default_rng(seed)
    y = df[label_col].to_numpy()
    return df.assign(**{name: rng.poisson(np.where(y == 1, rate + lift, rate))})


data_syn = (data
    .pipe(add_flag, "notes", "synthetic_i", 0.3, 0.4, 1)
    .pipe(add_category, "notes", "synthetic_lt", [0.05, 0.10, 0.20, 0.30, 0.35], 3)
    .pipe(add_score, "notes", "synthetic_s", 4, 4)
    .pipe(add_count, "notes", "synthetic_ip", 1.0, 0.8, 6))
