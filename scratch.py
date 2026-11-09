print(df_loss[KEY].isin(df_disq[KEY]).sum(), df_loss[KEY].head(5).tolist(), df_disq[KEY].head(5).tolist())
