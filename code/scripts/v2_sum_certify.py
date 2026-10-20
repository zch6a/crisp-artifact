import json, sys
d = json.load(open(sys.argv[1]))
fams = sys.argv[2].split(",") if len(sys.argv) > 2 else None
print(f"{'scorer':8s} {'cfg':10s} {'fam':8s} eps  | A cov   Cbar  tok | B cov  worstDB      | C cov  Cbar")
for r in d["rows"]:
    if fams and r["family"] not in fams:
        continue
    A, B, C = r["A"], r["B"], r["C"] or {}
    w = r["B"]["worst_db"]
    print(f"{r['scorer']:8s} {r['cfg']:10s} {r['family']:8s} {r['eps']:.2f} | {A['cov']:.4f} {A['cbar']:5.1f} {A['tok']:4.0f} | "
          f"{B['cov']:.4f} {w[0][:10]:10s}{w[1]:.3f} | {C.get('cov', float('nan')):.4f} {C.get('cbar', float('nan')):5.1f}")
