# Released results and flags

## JSON files

| File | Content | Paper |
|---|---|---|
| `certify_dev.json` | realized coverage vs target, per budget family and eps | Fig. 3 |
| `cert2seed32_k16.json`, `cert2seed32p10_k10.json` | certified-selection cross-fit results (k=16 / k=10) | Table 4 |
| `cert232_r2_k16.json`, `cert232p10_r2_k10.json` | per-question fallback depth r2, size, pool coverage | Table 4, label efficiency |
| `topk_curve.json` | EX gain vs candidate-pool size | Fig. 4 |
| `scale_curve.json` | coverage / EX vs schema size | Fig. 6 |
| `spider_sets.json` | Spider certificate sets | Spider experiment |
| `e2e_*.json` | per-generator EX and paired tests | Table 2 |
| `review_1019/` | threshold-family flags, warehouse cert2, two-stage flags | review response |

## flags/

Per-question correctness flags `{system: [1/0/null]}` in BIRD-Dev order (1534 entries; null =
failing gold or missing). The file names are the system tags used by the scripts; the main ones are
`flags_full_*` (full schema), `flags_cert2_*` (certificate + two-stage), `flags_cert2seed32_lam*`
(fallback depths), `flags_topk_single_*`, `flags_topk_tab32_*`, `flags_rsl*`, `flags_codes_*`.

To recompute a paired comparison, load two lists, drop indices where either is null, and bootstrap
the per-question difference (see `v2_pooled_cluster.py`).


## Quick check

```bash
python code/scripts/three_way.py results/ results/flags/
```
reproduces the certified-selection rows of Table 4:
pool@95 eps=.05 → coverage .9560, 17.7 columns, EX +2.29;
pool@98 eps=.05 → .9597, 18.3 columns, +2.03;
pool@95 eps=.10 → +2.51; pool@98 eps=.10 → +2.66.
`flags_q7b.json`, `flags_l8b.json`, `flags_q3.json` are aliases of `flags_B4_*.json`
and contain the `full`, `tab32_top10` and `tab32_top16` entries expected by `three_way.py`.
