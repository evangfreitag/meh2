Recommendation

Use the AI classifier as a weekly review worklist alongside the existing severity model and spaCy flags. Don't use it as a filter that removes flags.

What we found

The AI doesn't work as a false-positive filter. It only sees the last week of notes, so removing a flag on its say-so isn't safe.
The AI does surface claims where the notes point to a severe aspect of the claim that the severity model missed or scored with low confidence.
In manual review of the top-ranked AI labels, most were correct (per-label results in the table).
On the highest-confidence positives, the AI agreed with the model at the rate shown in the agreement table. This is agreement between two systems, not accuracy.

Proposed use

Each week, produce three sheets from the AI output. Each shows the top 25 claims per label, sorted by label, then AI score, then predicted probability. Each row includes the AI rationale and note text, so reviewers can judge it themselves.

Emergent severe claims: the severity model predicted not severe, and the AI assigns a severity label. Ties break toward the highest predicted probability, the closest to flipping.
Understated severe claims: the severity model predicted severe with low confidence, and the AI assigns a high score. Ties break toward the lowest predicted probability. These are claims a reviewer would otherwise have to dig for.
Agreement: the severity model predicted severe with high confidence, and the AI assigns the same label. This is a sanity check on the top-rated claims.

Limitations

The AI only sees the last week of notes, so coverage is limited to that window.
Precision was measured on the top-ranked labels per class, not on a random sample, so it overstates performance across all AI labels.
Several classes have few rows, so their precision estimates are wide.
Recall isn't measured. We can't say how many severe claims the AI misses.
The severity model has no truth column, so we can't measure whether the surfaced claims were actually severe.

Next steps

Once enough weeks of output have accumulated, look at how the severity model's keyword flags relate to the AI labels.
