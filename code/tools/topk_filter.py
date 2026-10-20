"""Keep only the k in {4, 10, 16, 40} selector prompts (cost-reduced matched-budget curve)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import hashlib
from v2data import RES
R = os.path.join(RES, "")
need = set()
T = [json.loads(l) for l in open(R + "tasks_topk_single.jsonl", encoding="utf-8")]
# rebuild the link-prompt hashes for the kept k from the dump order: link prompts were written per k in KS order
import importlib.util
KEEP = {4, 10, 16, 40}
spec = importlib.util.spec_from_file_location("tp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "v2_topk_pools.py"))
