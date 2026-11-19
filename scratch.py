h = hospital_mentions(latest, "notes_text")
print(h.count(), "of", latest.count(), "claims mention a hospital")

ex = extract_hospital(h, "notes_text")
ex.select(F.count("*").alias("n"),
          F.count("hospital_admit_date").alias("admit"),
          F.count("hospital_discharge_date").alias("discharge"),
          F.count("days_in_hospital").alias("days")).show()
