# Released reader scores

Each `.jsonl` line is `{"id": <question id>, "s": {"table|column": score, ...}}`.
Load them with `v2data.read_scores(path)`; the keys are the pairs returned by
`Example.schema.columns()`. All scores were produced with the reader versions below and are
released so that every certificate can be recomputed offline.

| File | Reader | Used for |
|---|---|---|
| `jev_t10_dev.jsonl` | Jev 1.13.0 (System-One API) | BIRD-Dev certificates, Tables 2/4, Fig. 3 |
| `jev_t10_holdout.jsonl` | Jev 1.13.0 | BIRD-train holdout, zero-Dev-label protocol |
| `jev_t10_pool.jsonl` | Jev 1.13.0 | reader distillation pool |
| `jev_t10_devabbr.jsonl` | Jev 1.13.0 | schema-abbreviation robustness |
| `jev_t10_spider.jsonl`, `jev_t10_spider2*.jsonl` | Jev 1.13.0 | Spider / Spider 2.0 transfer |
| `jev_t10_scale500.jsonl`, `jev_t10_scale2000.jsonl` | Jev 1.13.0 | scale experiments |
| `block_t10_dev_s7.jsonl` | distilled ModernBERT block reader (seed 7) | fully open pipeline |
| `block_t10_devabbr_s7.jsonl` | distilled block reader | abbreviation test |
| `laya_t10_dev.jsonl` | laya reader | reader ablation |
| `codes_dev.jsonl` | CodeS cross-encoder | cross-encoder baseline |
| `dist_t10_dev.jsonl` | distilled reader variant | ablation |
| `blockall.jsonl`, `block.jsonl` | distilled block reader on SNAILS warehouse | warehouse / SNAILS |
| `bird_bm25.jsonl`, `bird_embedding.jsonl` | tier-0 BM25 + MiniLM caches for BIRD-Dev | lexical fusion, all BIRD tables |
| `train_bm25.jsonl`, `train_emb.jsonl` | tier-0 caches for the BIRD-train holdout | zero-Dev-label protocol |
| `bird_spider_bm25.jsonl`, `lex_embedding_byid.jsonl` | tier-0 caches for Spider | Spider transfer |
| `bird_layapareto_laya_ft_bird_vask_L1024_t6.jsonl` | laya reader (tier-1) | reader ablation |

**Important:** the scripts read these files from `CRISP_CACHE`. For the released artifact set
`export CRISP_CACHE="$PWD/scores"` (the default is `cache/`, used for newly generated scores).

The Jev reader is a commercial API; the scores above are the released outputs, so no key is
needed to reproduce the paper. The distilled reader weights are in `assets/README.md`.
