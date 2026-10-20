"""Tier-1 scoring with the real Jev System-One API (POST /v1/systemone, shared state, noul probs).

Key is read from stdin (never written to disk / printed).  Resumable cache, thread pool, retries on
502/timeouts, hard stop on HTTP 402 or when the cumulative spend (from returned usage, $0.042/M
input tokens) reaches --budget-usd.  Same state and question template as the laya reader.
"""
import argparse, json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import CACHE, load_examples
from jevschema.risk import VIEWS
from jevschema.summary import SummaryConfig, bm25_scores, build_state

# Jev System-One endpoint = <CRISP_API_BASE>/systemone, e.g. CRISP_API_BASE=https://<jev-host>/v1
URL = os.environ.get("CRISP_API_BASE", "").rstrip("/") + "/systemone"
PRICE = 0.042 / 1e6
USAGE = os.path.join(CACHE, "jev_usage.json")
lock = threading.Lock()
GATE = "bm25"


def gate(e, top_t):
    pairs = e.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = e.question + (" " + e.evidence if e.evidence else "")
    best = {}
    for (t, c), v in zip(pairs, bm25_scores(q, docs)):
        best[t] = max(best.get(t, -1e9), v)
    return {t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_t]}


def post(key, payload, retries=8):
    data = json.dumps(payload).encode("utf-8")
    for att in range(retries):
        req = urllib.request.Request(URL, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            body = ex.read().decode("utf-8", "ignore")[:200]
            if ex.code == 402:
                raise SystemExit(f"HTTP 402 payment required: {body}")
            if ex.code in (400, 401, 403, 404, 422):
                raise RuntimeError(f"HTTP {ex.code}: {body}")
        except Exception:
            pass
        time.sleep(min(30, 2 * (att + 1)))
    raise RuntimeError("retries exhausted")


def score_one(key, e, model, top_t, chunk):
    state, _ = build_state(e.question_block(), e.schema, SummaryConfig())
    if GATE == "fk":
        from jevschema.gating import fk_gate
        keep = fk_gate(e)
    else:
        keep = gate(e, top_t)
    cols = [(t, c) for (t, c) in e.schema.columns() if t in keep]
    s, itok, otok, t0 = {}, 0, 0, time.time()
    for i in range(0, len(cols), chunk):
        part = cols[i:i + chunk]
        qs = {f"q{j}": {"type": "noul", "instructions": VIEWS["ask"].format(t=t, c=c)} for j, (t, c) in enumerate(part)}
        resp = post(key, {"model": model, "state": state, "questions": qs})
        ans = resp.get("answers") or {}
        for j, (t, c) in enumerate(part):
            a = ans.get(f"q{j}")
            if not isinstance(a, dict) or a.get("noul") is None:
                raise RuntimeError(f"bad answer q{j}: {str(resp)[:200]}")
            s[f"{t}|{c}"] = float(a["noul"])
        u = resp.get("usage") or {}
        itok += int(u.get("input_tokens", 0)); otok += int(u.get("output_tokens", 0))
    return {"id": e.ex_id, "s": s, "n_seq": (len(cols) + chunk - 1) // chunk, "in_tok": itok, "out_tok": otok,
            "enc_tokens": itok, "wall_s": round(time.time() - t0, 2), "model": resp.get("model", model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gate", default="bm25", choices=["bm25", "fk"])
    ap.add_argument("--min-tables", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--top-tables", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    a = ap.parse_args()
    key = sys.stdin.readline().strip()
    global GATE
    GATE = a.gate
    path = os.path.join(CACHE, (f"jev_t{a.top_tables}_{a.split}" if a.gate == "bm25" else f"jev_fk_{a.split}") + (f"_{a.tag}" if a.tag else "") + ".jsonl")
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    exs = load_examples(a.split)
    exs = exs[:: a.stride]
    exs = [e for e in exs if len(e.schema.tables) >= a.min_tables]
    exs = exs[: a.n] if a.n else exs
    todo = [e for e in exs if e.ex_id not in done]
    spent = json.load(open(USAGE)) if os.path.exists(USAGE) else {"in_tok": 0, "out_tok": 0, "usd": 0.0}
    print(f"split={a.split} todo={len(todo)}/{len(exs)} spent so far ${spent['usd']:.4f}", flush=True)
    t_end = time.time() + 60 * a.max_minutes
    n_ok = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        futs = {}
        it = iter(todo)
        def submit():
            e = next(it, None)
            if e is not None and time.time() < t_end and spent["usd"] < a.budget_usd:
                futs[pool.submit(score_one, key, e, a.model, a.top_tables, a.chunk)] = e
        for _ in range(a.workers):
            submit()
        while futs:
            for fu in as_completed(list(futs)):
                e = futs.pop(fu)
                try:
                    rec = fu.result()
                except SystemExit as ex:
                    print(str(ex), flush=True); t_end = 0; continue
                except Exception as ex:
                    print(f"  fail {e.ex_id}: {str(ex)[:160]}", flush=True); submit(); break
                with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    spent["in_tok"] += rec["in_tok"]; spent["out_tok"] += rec["out_tok"]
                    spent["usd"] = spent["in_tok"] * PRICE
                    json.dump(spent, open(USAGE, "w"))
                n_ok += 1
                if n_ok % 20 == 0:
                    print(f"  {n_ok} done, spent ${spent['usd']:.4f}, last in_tok={rec['in_tok']} wall={rec['wall_s']}s", flush=True)
                submit()
                break
    print(f"session done: {n_ok} scored, total spent ${spent['usd']:.4f} ({spent['in_tok']} in tok)", flush=True)


if __name__ == "__main__":
    main()
