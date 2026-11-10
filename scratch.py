The model gives each claim a Fast Track probability. The cutoff is the probability above which we auto-route a claim to Fast Track.
We picked the cutoff on a separate validation slice so that about 90% of claims routed to Fast Track really are Fast Track.
We set precision high on purpose. A claim wrongly sent to Fast Track gets less attention than it needs. A Fast Track claim that’s missed just goes through the normal queue, which is how everything works today.
The tradeoff is recall. A high cutoff means the model only routes the claims it’s most sure about, so it catches fewer of the true Fast Track claims (about 56% in your run).
The cutoff is adjustable. Lowering it routes more claims to Fast Track, at the cost of more false positives, and the business can pick the balance.
It was chosen on validation and applied unchanged to test, so test precision is an honest check.
