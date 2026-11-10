bad = chk[chk["gross_total_silver"] != chk["gross_total_test"]]
print(bad["gross_total_silver"].isna().sum(), bad["gross_total_test"].isna().sum())
print((bad["gross_total_silver"] - bad["gross_total_test"]).describe())
print(bad.head(10))
