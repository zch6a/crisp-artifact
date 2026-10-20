"""Select the prompts of given systems from a dump (prompts_all2 + map) -> smaller file for an expensive generator."""
import json, sys
dump, mapf, systems, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
M = json.load(open(mapf))["systems"]
need = set()
for s in systems:
    need |= set(M[s])
n = 0
with open(out, "w", encoding="utf-8") as f:
    for l in open(dump, encoding="utf-8"):
        if json.loads(l)["h"] in need:
            f.write(l); n += 1
print(f"{n} prompts for {systems}")
