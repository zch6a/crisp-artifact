"""Keep only prompts whose md5 is not yet in a generation cache: v2_delta.py <prompts.jsonl> <cache.jsonl> <out.jsonl>."""
import json, sys
inp, cache, out = sys.argv[1:4]
have = {json.loads(l)["h"] for l in open(cache, encoding="utf-8")}
n = 0
with open(out, "w", encoding="utf-8") as f:
    for l in open(inp, encoding="utf-8"):
        if json.loads(l)["h"] not in have:
            f.write(l); n += 1
print(f"{n} new prompts -> {out}")
