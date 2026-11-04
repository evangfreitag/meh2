import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

FLAG_COLS = [c for c in df.columns if c.startswith("flag_")]  # your 0/1 flag columns; adjust the filter
MIN_COUNT = 30   # skip flags that fire on fewer claims than this
CUT = 0.7        # 1 = Jaccard distance; lower makes tighter, smaller groups

X = csr_matrix(df[FLAG_COLS].fillna(0).to_numpy() > 0, dtype=np.float32)
sizes = np.asarray(X.sum(axis=0)).ravel()
keep = sizes >= MIN_COUNT
X, names, sizes = X[:, keep], np.array(FLAG_COLS)[keep], sizes[keep]

# Jaccard: claims where both fire, divided by claims where either fires
both = (X.T @ X).toarray()
dist = 1 - both / (sizes[:, None] + sizes[None, :] - both)
np.fill_diagonal(dist, 0)

groups = fcluster(linkage(squareform(dist, checks=False), method="average"), t=CUT, criterion="distance")
out = pd.DataFrame({"group": groups, "flag": names, "claims": sizes.astype(int)}).sort_values(["group", "claims"], ascending=[True, False])
print(out.groupby("group").size().describe())
print(out.to_string(index=False))
