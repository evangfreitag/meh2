import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

LABEL_COL = "cost_flag"
TOP_N = 25
SAMPLE = 3000

data = VAL.sample(min(SAMPLE, len(VAL)), random_state=0)
words = words_and_chars.transform(data[TEXT_COL])
S = struct_features(data)[list(S_train.columns)]
X = S.assign(**{f"text_{t}": text_model.predict_proba(words)[:, i] for i, t in enumerate(text_model.classes_)})[list(C_train.columns)]
y = data[LABEL_COL]

imp = permutation_importance(combined_model, X, y, scoring="roc_auc", n_repeats=3, random_state=0)

order = imp.importances_mean.argsort()[::-1][:TOP_N][::-1]
plt.figure(figsize=(8, 10))
plt.barh(np.array(X.columns)[order], imp.importances_mean[order])
plt.xlabel("Drop in ROC AUC when the variable is shuffled")
plt.title(f"Top {TOP_N} variables, combined model")
plt.tight_layout()
plt.show()
