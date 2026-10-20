"""Build the flags directory for three_way.py (fully open pipeline, TAGS=d32,d32p10).
Per-depth flags: flags_tw_<g>.json (systems cert2seedd32_lam<l>) -> flags_cert2d32_lam<l>_<g>.json (one list each).
Full schema: q7b / l8b from the original fully open run (flags_D_<g>.json, same ollama generations as the paper);
q3 from the regeneration (same vLLM engine as its per-depth generations).
  python tw_flags.py <dir with flags_tw_*.json and flags_D_*.json> <out dir>"""
import json, os, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for g in ("q7b", "l8b", "q3"):
    tw = json.load(open(os.path.join(src, f"flags_tw_{g}.json")))
    for s, v in tw.items():
        if s.startswith("cert2seedd32_lam"):
            lam = s.split("lam")[1]
            json.dump(v, open(os.path.join(out, f"flags_cert2d32_lam{lam}_{g}.json"), "w"))
    full = tw["full"] if g == "q3" else json.load(open(os.path.join(src, f"flags_D_{g}.json")))["full"]
    json.dump({"full": full}, open(os.path.join(out, f"flags_{g}.json"), "w"))
    ex = lambda v: 100 * sum(x for x in v if x is not None) / sum(1 for x in v if x is not None)
    print(g, "full EX %.2f" % ex(full), {s: round(ex(v), 2) for s, v in tw.items() if s != "full"})
