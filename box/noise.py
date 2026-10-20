import json, subprocess, sys
import numpy as np
sel = json.load(open("tasks_noise_sel.json")); idx = set(sel["idx"])
rows = [json.loads(l) for l in open("tasks_all2.jsonl") if json.loads(l)["sys"] in sel["systems"] and json.loads(l)["i"] in idx]
open("tasks_noise.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
g = sys.argv[1]
subprocess.run(["python", "ex_remote.py", "tasks_noise.jsonl", f"gen_{g}_rep.jsonl", f"flags_noise_{g}.json", "48"], check=True, capture_output=True)
A = json.load(open(f"flags_all2_{g}.json")); B = json.load(open(f"flags_noise_{g}.json"))
for s in sel["systems"]:
    pr = [(A[s][i], B[s][i]) for i in sel["idx"] if A[s][i] is not None and B[s][i] is not None]
    flip = np.mean([a != b for a, b in pr]); ea = np.mean([a for a, _ in pr]); eb = np.mean([b for _, b in pr])
    print(f"{g} {s:18s} n={len(pr)} EX run1={ea:.4f} run2={eb:.4f} |diff|={abs(ea-eb):.4f} question-level flip rate={flip:.3f}")
d1 = [A["jev_t10:t1_e0.05"][i] - A["full"][i] for i in sel["idx"] if None not in (A["full"][i], A["jev_t10:t1_e0.05"][i], B["full"][i], B["jev_t10:t1_e0.05"][i])]
d2 = [B["jev_t10:t1_e0.05"][i] - B["full"][i] for i in sel["idx"] if None not in (A["full"][i], A["jev_t10:t1_e0.05"][i], B["full"][i], B["jev_t10:t1_e0.05"][i])]
print(f"{g} CRISP-full diff on the 300 questions: run1 {np.mean(d1):+.4f}  run2 {np.mean(d2):+.4f}")
