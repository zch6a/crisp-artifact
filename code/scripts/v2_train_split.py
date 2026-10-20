"""BIRD-train split for v2.

laya_ft_bird was fine-tuned on load_bird_train(limit=3500) (train 3000 + eval 500, shuffled).
Everything after the first 3500 kept examples was never seen by laya. We keep only holdout
examples whose db_id is disjoint from the laya pool, so tier-1 scores on them are honest.
"""
import json, os, sys, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.bird_train import load_bird_train
from jevschema.paths import RESULTS

t0 = time.time()
exs, st = load_bird_train(limit=None, log_every=0)
print("all kept:", len(exs), st, f"{time.time()-t0:.1f}s")
pool, rest = exs[:3500], exs[3500:]
pool_dbs = {e.db_id for e in pool}
hold = [e for e in rest if e.db_id not in pool_dbs]
print("laya pool:", len(pool), "dbs", len(pool_dbs))
print("rest:", len(rest), "holdout (db-disjoint):", len(hold), "dbs", len({e.db_id for e in hold}))
ncols = [e.schema.n_columns() for e in hold]
ngold = [len(e.gold.requirements) for e in hold]
import numpy as np
print("holdout cols mean/median/max:", np.mean(ncols), np.median(ncols), max(ncols))
print("holdout gold mean:", np.mean(ngold))
out = {"pool_ids": [e.ex_id for e in pool], "holdout_ids": [e.ex_id for e in hold],
       "stats": {"n_all": len(exs), "n_pool": len(pool), "n_holdout": len(hold),
                 "pool_dbs": len(pool_dbs), "holdout_dbs": len({e.db_id for e in hold}),
                 "holdout_cols_mean": float(np.mean(ncols)), "holdout_gold_mean": float(np.mean(ngold))}}
os.makedirs(RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "bird_train_split.json"), "w"), indent=1)
print("saved")
