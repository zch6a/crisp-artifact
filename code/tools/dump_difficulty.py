"""Dump BIRD-Dev difficulty per question index (order of load_examples('dev')) -> results/dev_difficulty.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from v2data import RES, load_examples
d = [e.difficulty or "unknown" for e in load_examples("dev")]
print(len(d), {k: d.count(k) for k in set(d)})
json.dump(d, open(os.path.join(RES, "dev_difficulty.json"), "w"))
