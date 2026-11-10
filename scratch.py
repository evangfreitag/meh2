chk = silver_pd.merge(test[["claim_number", "gross_total"]], on="claim_number", suffixes=("_silver", "_test"))
print((chk["gross_total_silver"] != chk["gross_total_test"]).sum(), "of", len(chk))
