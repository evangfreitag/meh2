from pyspark.sql import Window, functions as F

import numpy as np
import pandas as pd

r = permutation_importance(...)   # whatever you named it
importance_df = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
