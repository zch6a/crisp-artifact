# CRISP: Certified Schema Pruning for Text-to-SQL

Artifact for the PVLDB submission **CRISP: Certified Schema Pruning for Text-to-SQL** (v19b).
CRISP turns schema pruning into a certified operation: a reader ranks columns, a monotone
key/foreign-key closure makes the pruned schemas nested, split-conformal calibration picks a
budget with a question-level coverage guarantee, and a second calibrated fallback certifies the
schema that an LLM selector finally hands to the generator.

## What is in this repository

```
code/
  src/jevschema/        library: schema graph, readers, closure, certificate, selection, EX
  scripts/              experiment drivers (v2_*.py, three_way.py, crisp_nb.py, nb_gen.py, ...)
  tools/                prompt/task builders for specific runs
  configs/              jev_api.json.example (no key is stored in the repository)
box/                    GPU-side generation and EX evaluation (vLLM / ollama / OmniSQL / Spider 2.0)
scores/                 released reader outputs (Jev, laya, block reader, CodeS) — no API needed
results/                result JSONs and per-question flags behind the paper's tables
reproduce/              the review-1019 analyses (threshold family, label efficiency, parsing)
data/                   (not shipped) put downloaded datasets here; see "Data" below
```

The reader scores in `scores/` are sufficient to recompute every certificate and coverage table
in the paper without calling any model. The per-question flags in `results/flags/` are sufficient
to recompute the EX comparisons. Generation scripts are provided for re-running models.

## Setup

Python 3.12.

```bash
pip install -r code/requirements.txt
# on a GPU machine for generation:  pip install vllm
# for local generation without vLLM: install ollama (https://ollama.com)
```

Scripts add `code/src` to `sys.path` themselves; no install step is needed.

### Five-minute check (no GPU, no API)

The released reader scores are in `scores/`, and the scripts read them through `CRISP_CACHE`:

```bash
export CRISP_CACHE="$PWD/scores"
export CRISP_DATA="$PWD/data"
export CRISP_RESULTS="$PWD/results"
# requirement extraction needs sqlglot (pinned in code/requirements.txt);
# after placing BIRD dev.json/dev_tables.json in data/raw/bird/:
python reproduce/certificate_dev.py --eps 0.05
# expected: coverage 0.955, 20.2 columns (no holdout/train annotations needed)
python reproduce/threshold_family.py --eps 0.05 --out results/
# expected: coverage 0.9505, 16.8 columns, plus the two output files
```

Then reproduce Table 4 from the released flags without any model call:

```bash
python code/scripts/three_way.py results/ results/flags/
```

## Data

| Dataset | Where to get it | Place it at (default) |
|---|---|---|
| BIRD dev (questions, `dev_tables.json`, sqlite databases) | https://bird-bench.github.io/ (dev.zip) | `data/raw/bird/dev.json`, `data/raw/bird/dev_tables.json`, `data/bird_dev/dev_databases/<db>/<db>.sqlite` |
| BIRD train with linking annotations (`train_0711.jsonl`) | Hugging Face `arcwise/bird_sql_annotated` | `data/raw/bird/arcwise/train_0711.jsonl` |
| Spider dev/train | https://yale-lily.github.io/spider (or HF `xlangai/spider`) | `data/raw/spider_*.parquet`, `spider_schema.json` |
| Spider 2.0-lite | https://github.com/xlang-ai/Spider2 (`spider2-lite/`) | `ext/Spider2/spider2-lite/`; run `box/spider2_extract.py`, `box/spider2_gold2.py` |
| SNAILS (8 real databases, 396 questions; 394 with executable reference SQL) | SNAILS benchmark (Luoma and Kumar, PACMMOD 2025) | `data/snails/*.sqlite`, `data/snails/pairs/*.sql` |
| OmniSQL released inputs | HF `seeklhy/OmniSQL-datasets` (`data.zip`) | `ext/omnidata/data/...`; copy `dev_bird.json` to `data/omnisql_dev_bird.json` |
| OmniSQL-7B | HF `seeklhy/OmniSQL-7B` | `models/OmniSQL-7B` |
| RSL-SQL released linkings | RSL-SQL repo, `main_result/BIRD_DEV/{gpt-4o,deepseek}` | `data/rsl/<model>_LLM.json`, `data/rsl/<model>_augmentation.json` |
| MiniLM embeddings (tier-0) | HF `sentence-transformers/all-MiniLM-L6-v2` | `models/miniLM` |

SNAILS and the 2,635-column warehouse are built from the SNAILS release; see
`code/scripts/crisp_nb.py` (`build`, `reader`, `cert`, `cert2`, `eval`).

## Released scores and weights

- `scores/` contains the reader outputs used by the paper (Jev 1.13.0, laya, the distilled
  ModernBERT block reader, and the CodeS cross-encoder). With these files, all certificates can be
  recomputed without querying a model and without the commercial Jev API.
- The distilled block-reader weights (~1.6 GB) are not stored in git. They will be released
  together with the formal publication of the paper. To score new schemas with the local reader,
  unpack them to `models/laya_block_distill`.
- Jev is a commercial System-One API; the repository never stores a key. `code/configs/jev_api.json.example`
  shows the configuration. The paper's scores were produced with version 1.13.0 and are released
  above so that the guarantees can be checked offline.

## Reproducing the paper

| Paper item | Script |
|---|---|
| Coverage / validity (Fig. 3), ablations (Table 8) | `code/scripts/v2_certify.py`, `v2_validity.py`, `v2_ablation_cert.py`, `v2_ncal.py`, `v2_dbcond.py`, `v2_mondrian.py` |
| Coverage across workloads (Table 1) | `code/scripts/v2_perturb_eval.py`, `v2_scale_curve.py`, `crisp_nb.py cert` |
| Main EX table (Table 2) | `code/scripts/v2_e2e.py`; or recompute from `results/flags/` |
| OmniSQL official setting (Table 3) | `box/official_prune.py`, `box/omni_template*.py`, `box/ex_remote.py` |
| Certified selection (Table 4, Fig. 4) | `code/scripts/three_way.py results/ results/flags/`; pools: `v2_topk_pools.py` |
| Equal-coverage linker baselines (Table 5) | `code/scripts/v2_rsl_eval.py`, `v2_rsl_eval_lib.py`, `v2_codes_eval.py` |
| Cost (Table 6, Fig. 5) | `code/scripts/v2_cost.py`, `v2_pareto_cost.py` |
| Scale (Fig. 6) | `code/scripts/scale_build.py`, `v2_scale_curve.py`, `v2_scale_score.py` |
| SNAILS + warehouse (Fig. 7) | `code/scripts/crisp_nb.py` (`build/reader/cert/cert2/eval`), `nb_gen.py`, `nb_pooled.py` |
| Serving latency (Table 7) | `code/scripts/lat_bench.py`, `lat_reader.py` |
| Review-1019 analyses | `reproduce/README.md` |

All scripts print their exact arguments in the docstring. Use `--help` where available.
The released `results/` and `scores/` let a reviewer recompute the coverage and EX tables
without any GPU or API call.

### EX evaluation

`box/ex_remote.py` reads `DBDIR` and `DBFMT`. For BIRD-Dev the databases are nested, so use
`DBDIR=data/bird_dev/dev_databases DBFMT='{DB}/{db}/{db}.sqlite'`.

## Environment variables

All roots are defined in `code/src/jevschema/paths.py`:

| Variable | Default | Meaning |
|---|---|---|
| `CRISP_ROOT` | repository root | base for the defaults below |
| `CRISP_DATA` | `<root>/data` | datasets |
| `CRISP_CACHE` | `<root>/cache` | scorer/generator caches (set to `scores` to use the released scores) |
| `CRISP_RESULTS` | `<root>/results` | result JSONs, prompt/task files |
| `CRISP_EXT` | `<root>/ext` | external releases (OmniSQL, Spider 2.0) |
| `CRISP_MODELS` | `<root>/models` | local checkpoints |
| `CRISP_BOX` | `<root>/box` | GPU-side working directory |
| `BIRD_DB_DIR` | `<data>/bird_dev/dev_databases` | BIRD-Dev sqlite files |

Legacy `JEVSCHEMA_*` variables still override the corresponding paths.

## License

MIT, see `LICENSE`. The released reader scores are outputs of the models listed above; the
datasets keep their original licenses.
