"""Print real per-question decision records (question 0 and one loss case) for the proposal's log example."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import argparse
from v2data import BOX, RES, load_examples
from v2_e2e import build_systems
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
H = json.load(open(os.path.join(RES, "hybridv_g6l_sets.json")))["sets"]
F = json.load(open(os.path.join(BOX, "flags_offgreedy_all_set.json")))
for i in (0, 7):
    e = dev[i]
    print(json.dumps({"q": i, "db": e.db_id, "difficulty": e.difficulty, "n_cols": len(e.schema.columns()),
                      "cert_cols": len(S["t1_e0.05"][i]), "cert_covers_gold": set(map(tuple, e.gold.flat_pairs())) <= set(map(tuple, S["t1_e0.05"][i])),
                      "selected": H[i], "ex_full": F["off:full"][i], "ex_crisp": F["off:jev_t10:t1_e0.05"][i], "ex_two_stage": F["off:hybridv_g6l"][i]}, ensure_ascii=False))
