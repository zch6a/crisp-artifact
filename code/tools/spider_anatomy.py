"""Spider-Dev official-input wins/losses of certified sets vs full, split by gold coverage."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from v2data import RES, load_examples
S = json.load(open(os.path.join(RES, "spider_sets.json")))
F = json.load(open(os.path.join(RES, "flags_q20_spider_set.json")))
ex = {e.ex_id: e for e in load_examples("spider")}
for k in ("jev_t10:t1_e0.05", "jev_t10:t1_e0.1", "t0_e0.05"):
    rows = {"cov": [0, 0, 0], "miss": [0, 0, 0]}
    for i, eid in enumerate(S["ids"]):
        g = {(t.lower(), c.lower()) for t, c in ex[eid].gold.flat_pairs()}
        kept = {(t.lower(), c.lower()) for t, c in S["sets"][k][i]}
        part = "cov" if g <= kept else "miss"
        a, b = F["sp:full"][i], F["sp:" + k][i]
        rows[part][0] += 1; rows[part][1] += int(b > a); rows[part][2] += int(b < a)
    print(k, {p: f"{v[0]} q w/l {v[1]}/{v[2]}" for p, v in rows.items()})
