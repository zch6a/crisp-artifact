"""Holm-Bonferroni over the family of *secondary* EX claims (the primary hypothesis — CRISP eps=0.05 pooled over
generators — is tested alone at alpha=0.05).  p-values: question-clustered pooled bootstrap for pooled claims,
paired bootstrap for single-generator claims (values produced by pooled_flags.py / paired_flags.py today)."""
fam = [
    ("pooled 5 gens: CRISP eps=0.1 > full", 0.0130),
    ("pooled 5 gens: CRISP eps=0.02 > full", 0.0622),
    ("pooled 5 gens: open laya reader eps=0.05 > full", 0.0284),
    ("confirmatory GPT+DS: CRISP eps=0.05 > full", 0.0082),
    ("32B: hybrid (32B linker) > full", 0.095),
    ("32B: hybrid (distilled pool) > full", 0.187),
    ("hard bench 32B: hybrid > raw full", 0.2524),
    ("OmniSQL: CRISP + DeepSeek selection > full", 0.004),
    ("OmniSQL: CRISP + GPT selection > full", 0.002),
    ("OmniSQL hard bench: hybrid > raw full", 0.008),
    ("OmniSQL hard bench: canonical full > raw full", 0.016),
    ("OmniSQL SC: CRISP + DeepSeek selection > SC full", 0.0002),
    ("mechanism 32B: random context > CRISP (3 seeds)", 0.0056),
    ("mechanism OmniSQL: random context > CRISP", 0.0002),
    ("DeepSeek alone: CRISP eps=0.05 > full", 0.0204),
]
m = len(fam)
order = sorted(range(m), key=lambda i: fam[i][1])
adj, running = [0.0] * m, 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * fam[i][1]))
    adj[i] = running
print(f"{'claim':52s} {'p':>7s} {'Holm p':>7s}  sig@.05")
for i in order:
    print(f"{fam[i][0]:52s} {fam[i][1]:7.4f} {adj[i]:7.4f}  {'yes' if adj[i] < 0.05 else 'no'}")
