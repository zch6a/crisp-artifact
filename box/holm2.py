"""Holm-Bonferroni within research-question families (families defined after the fact - stated as such), using the
p-values produced today by pooled_*.py / paired_flags.py / strata_pooled*.py (bootstrap, 10k or 5k resamples).
The primary hypothesis (CRISP eps=0.05 pooled over the 5 pre-specified generators) is tested alone at alpha=0.05."""
FAM = {
 "F1 primary + confirmatory": [
  ("CRISP eps=.05, 5 generators pooled", 0.0014),
  ("CRISP eps=.05, confirmatory GPT-5.6 + DeepSeek", 0.0082)],
 "F2 method variants (our input, 5 gens pooled)": [
  ("CRISP eps=.1", 0.0044), ("CRISP eps=.02", 0.0622), ("open laya reader eps=.05", 0.0202)],
 "F3 two-stage, pooled": [
  ("two-stage, uniform DS linker, 5 gens", 0.0001), ("two-stage, 6 gens incl. GPT-6-Luna", 0.0001),
  ("zero-Jev two-stage, 4 gens", 0.0001), ("zero-Jev two-stage, 5 gens", 0.0001)],
 "F4 difficulty strata (two-stage / zero-Jev, challenging)": [
  ("two-stage challenging", 0.0446), ("zero-Jev challenging", 0.0194),
  ("two-stage moderate", 0.0002), ("zero-Jev moderate", 0.0024)],
 "F5 OmniSQL official setting (omnibus: 6 configs pooled per question, pooled_configs.py)": [
  ("official metric, greedy+SC x 3 linkers", 0.0052), ("multiset metric, greedy+SC x 3 linkers", 0.0494)],
 "F5b OmniSQL official setting, individual configs (official metric; reported for completeness)": [
  ("greedy two-stage DS", 0.074), ("greedy value-aware DS", 0.017), ("greedy value-aware GPT-6-Luna", 0.028),
  ("SC two-stage DS", 0.031), ("SC value-aware DS", 0.008), ("SC value-aware GPT-6-Luna", 0.040)],
 "F6 OmniSQL, our input (official parser)": [
  ("two-stage DS", 0.0005), ("zero-Jev two-stage", 0.003)],
 "F7 mechanism": [
  ("32B: random context > CRISP (3 seeds)", 0.0056), ("OmniSQL: random context > CRISP", 0.0002)],
}


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


allc = []
for fam, cl in FAM.items():
    adj = holm([p for _, p in cl])
    print(f"\n{fam}")
    for (n, p), a in zip(cl, adj):
        print(f"  {n:44s} p={p:.4f}  Holm={a:.4f}  {'sig' if a < 0.05 else 'n.s.'}")
    allc += cl[1:] if fam.startswith("F1") else ([] if fam.startswith("F5b") else cl)
adj = holm([p for _, p in allc])
print(f"\nGlobal Holm over all {len(allc)} secondary claims: {sum(a < 0.05 for a in adj)} remain significant")
for (n, p), a in sorted(zip(allc, adj), key=lambda x: x[1]):
    print(f"  {n:44s} p={p:.4f}  Holm={a:.4f}")
