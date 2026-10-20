"""Configurable filesystem roots.

Every root can be overridden with an environment variable; the defaults are relative to the
repository root (the directory that contains code/ and box/).

  CRISP_ROOT     repository root                       (default: computed from this file)
  CRISP_DATA     datasets (BIRD, Spider, Spider 2.0)   (default: <root>/data)
  CRISP_CACHE    scorer / generator caches             (default: <root>/cache)
  CRISP_RESULTS  result JSON / prompt + task files     (default: <root>/results)
  CRISP_EXT      external releases (OmniSQL data/repo, Spider2 repo)  (default: <root>/ext)
  CRISP_MODELS   local model checkpoints               (default: <root>/models)
  CRISP_BOX      working dir of the GPU-side scripts in box/ (flags_*.json, tasks_*.jsonl)
                                                       (default: <root>/box)
  BIRD_DB_DIR    BIRD-Dev sqlite databases <db>/<db>.sqlite
                                                       (default: <data>/bird_dev/dev_databases)
"""
import os

ROOT = os.environ.get("CRISP_ROOT", os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
DATA = os.environ.get("CRISP_DATA", os.path.join(ROOT, "data"))
CACHE = os.environ.get("CRISP_CACHE", os.path.join(ROOT, "cache"))
RESULTS = os.environ.get("CRISP_RESULTS", os.path.join(ROOT, "results"))
EXT = os.environ.get("CRISP_EXT", os.path.join(ROOT, "ext"))
MODELS = os.environ.get("CRISP_MODELS", os.path.join(ROOT, "models"))
BOX = os.environ.get("CRISP_BOX", os.path.join(ROOT, "box"))
BIRD_DB_DIR = os.environ.get("BIRD_DB_DIR", os.path.join(DATA, "bird_dev", "dev_databases"))
