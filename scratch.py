import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def single_column_auc(X_train, y_train, X_test, y_test, cols):
    rows = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=30)
            m.fit(X_train[[col]], y_train)
            p = m.predict_proba(X_test[[col]])[:, 1]
        else:
            rate = y_train.groupby(X_train[col].fillna("missing")).mean()
            p = X_test[col].fillna("missing").map(rate).fillna(y_train.mean())
        rows.append((col, roc_auc_score(y_test, p)))
    for name, auc in sorted(rows, key=lambda r: -r[1])[:15]:
        print(f"{name}\t{auc:.3f}")

single_column_auc(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, cols=X_train.columns)
