tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

"n_test": len(y_test),
"actual_pos": int(y_test.sum()),
"true_pos": int(tp),
"fpr": round(fp / (fp + tn), 4),
