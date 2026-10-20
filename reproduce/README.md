# Reproducing the review-1019 analyses

These scripts use the released reader scores and lexical caches in `scores/`; no API calls are needed.
Set `export CRISP_CACHE="$PWD/scores"` (and `CRISP_DATA` / `CRISP_RESULTS`) before running them.
For generation, which writes to `CRISP_CACHE`, copy the released scores to a writable directory first:
`mkdir -p cache && cp scores/* cache/ && export CRISP_CACHE="$PWD/cache"`.

## Dev-only certificate (main validity number)
```bash
python reproduce/certificate_dev.py --eps 0.05
# expected: coverage 0.955, 20.2 columns; no holdout or train annotations needed
```

## Strongest single-stage baseline (threshold family)
```bash
python reproduce/threshold_family.py --eps 0.05 --out results/
# generates results/thresh_e0.05_genprompts.jsonl and tasks_thresh_e0.05.jsonl
```
Generate with the same engines as the paper (ollama qwen2.5-coder:7b / llama3.1:8b on one GPU;
Qwen3-8B with vLLM), then evaluate:
```bash
DBDIR=data/bird_dev/dev_databases DBFMT='{DB}/{db}/{db}.sqlite' \
  python box/ex_remote.py results/tasks_thresh_e0.05.jsonl cache/gen_thresh_q7b.jsonl flags_thresh_q7b.json 32
```
The paper's numbers are in `results/review_1019/` (q7b +1.31, l8b +2.22, q3 +1.24; pooled +1.59).

## Label efficiency of the certified fallback
```bash
python reproduce/label_efficiency.py --r2 results/cert232_r2_k16.json --eps 0.05
```

## Three-way certified selection
```bash
# after generating the selector outputs {h,text} and the per-depth generations:
python code/scripts/three_way.py results/ results/flags/
```
See `code/scripts/three_way.py` for the expected file names
(`cert232_r2_k16.json`, `cert232p10_r2_k10.json`, `flags_*.json`).

## Parsing vLLM outputs
```bash
python reproduce/parse_text.py --src cache/gen_tag.jsonl --dst cache/gen_tag_sql.jsonl
```
