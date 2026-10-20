"""Dump deduplicated generation prompts of every EX system (for an external generator)."""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import RES, load_examples
from v2_e2e import build_systems, prompt_of

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(RES, "prompts_all.jsonl")); ap.add_argument("--samples", type=int, default=0); ap.add_argument("--tier1s", default="jev_t10,laya_t10"); a0 = ap.parse_args()
import v2_e2e
v2_e2e.SAMPLES = a0.samples
dev, hold = load_examples("dev", samples=a0.samples), load_examples("holdout")
seen, rows, sysmap = set(), [], {}
for tier1 in a0.tier1s.split(","):
    a = argparse.Namespace(tier1=tier1, beta=0.02, route_frac=0.4)
    S, _ = build_systems(a, dev, hold)
    for name, kept in S.items():
        key = name if name in ("full", "oracle") or name.startswith("t0") else f"{tier1}:{name}"
        if key in sysmap:
            continue
        hs = []
        for e, k in zip(dev, kept):
            p = prompt_of(e, k); h = hashlib.md5(p.encode("utf-8")).hexdigest(); hs.append(h)
            if h not in seen:
                seen.add(h); rows.append({"h": h, "prompt": p})
        sysmap[key] = hs
with open(a0.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"ids": [e.ex_id for e in dev], "systems": sysmap}, open(a0.out.replace(".jsonl", "_map.json"), "w"))
print(f"{len(rows)} unique prompts, systems: {list(sysmap)}")
