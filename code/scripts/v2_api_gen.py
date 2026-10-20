"""SQL generation with an OpenAI-compatible chat API (strong generators: GPT-5.6 Luna, DeepSeek-V4 Flash).

  key is read from stdin (never written to disk).
  v2_api_gen.py --prompts results/prompts_api.jsonl --model gpt-5.6-luna --tag gpt56luna [--limit 20] [--workers 128]
  -> cache/gen_<tag>.jsonl  ({h, sql}; same format as the other generator caches, so v2_e2e --cache-only works)
  -> cache/gen_<tag>_usage.json (token usage)
Adaptive concurrency (AIMD): the in-flight limit grows by one per ~limit successes and is cut by 30% on HTTP 429,
so the run settles at the gateway's per-key concurrency limit. 429s are retried without counting as failures.
"""
import argparse, json, os, random, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import CACHE
from jevschema.generator import extract_sql

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--base", default=os.environ.get("CRISP_API_BASE", ""),
                help="OpenAI-compatible base URL, e.g. https://<gateway>/v1 (default: env CRISP_API_BASE)")
ap.add_argument("--workers", type=int, default=128, help="max in-flight requests (upper bound for AIMD)")
ap.add_argument("--start", type=float, default=16, help="initial in-flight limit")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--max-minutes", type=float, default=0)
ap.add_argument("--reverse", action="store_true", help="process prompts from the end (second key working the same list)")
ap.add_argument("--skip-file", default="", help="other worker's output; its hashes are skipped (re-read every 60 s)")
ap.add_argument("--raw", action="store_true", help="store raw text as {h, text} (linking prompts) instead of extracted SQL")
ap.add_argument("--samples", type=int, default=1, help=">1: self-consistency samples (separate calls at --temperature)")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--stream", action="store_true",
                help="SSE streaming: keeps the connection alive for long reasoning (gateway cuts silent requests at ~100 s -> 502)")
a = ap.parse_args()
assert a.base, "set --base or the CRISP_API_BASE environment variable"
KEY = sys.stdin.readline().strip()
assert KEY, "key expected on stdin"

out_path = os.path.join(CACHE, f"gen_{a.tag}.jsonl")
use_path = os.path.join(CACHE, f"gen_{a.tag}_usage.json")
done = set()
if os.path.exists(out_path):
    for l in open(out_path, encoding="utf-8"):
        try:
            done.add(json.loads(l)["h"])
        except Exception:
            pass
U = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0, "fail": 0}
if os.path.exists(use_path):
    U.update(json.load(open(use_path)))
U["rate_limited"] = U.get("rate_limited", 0)
rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
todo = [r for r in rows if r["h"] not in done]
if a.limit:
    todo = todo[:a.limit]
if a.reverse:
    todo = todo[::-1]
SKIP, SKIP_T = set(), [0.0]


def skipped(h):
    if not a.skip_file:
        return False
    if time.time() - SKIP_T[0] > 60:
        SKIP_T[0] = time.time()
        if os.path.exists(a.skip_file):
            for l in open(a.skip_file, encoding="utf-8"):
                try:
                    SKIP.add(json.loads(l)["h"])
                except Exception:
                    pass
    return h in SKIP
print(f"{a.model}: {len(rows)} prompts, {len(done)} cached, {len(todo)} to do", flush=True)
lock = threading.Lock()
T0 = time.time()


class AIMD:
    def __init__(self, start, hi):
        self.lim, self.hi, self.inflight = float(start), hi, 0
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            while self.inflight >= int(self.lim):
                self.cv.wait()
            self.inflight += 1

    def release(self, ok, limited):
        with self.cv:
            self.inflight -= 1
            if limited:
                self.lim = max(2.0, self.lim * 0.7)
            elif ok:
                self.lim = min(self.hi, self.lim + 1.0 / max(1.0, self.lim))
            self.cv.notify_all()


G = AIMD(a.start, a.workers)


def call(prompt):
    req_d = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.stream:
        req_d.update(stream=True, stream_options={"include_usage": True})
    body = json.dumps(req_d).encode("utf-8")
    last, errs = "", 0
    while errs < 6:
        G.acquire()
        ok = limited = False
        try:
            req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=300) as r:
                if not a.stream:
                    d = json.loads(r.read().decode("utf-8"))
                    ok = True
                    return (d["choices"][0]["message"].get("content") or ""), d.get("usage") or {}
                parts, usage = [], {}
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ch = json.loads(data)
                    usage = ch.get("usage") or usage
                    for c in ch.get("choices") or []:
                        parts.append((c.get("delta") or {}).get("content") or "")
            ok = True
            return "".join(parts), usage
        except urllib.error.HTTPError as ex:
            last = f"HTTP {ex.code}: {ex.read()[:200]!r}"
            # 429 and gateway overload (the relay answers 502/503/504/52x under high concurrency) -> back off, retry
            limited = ex.code in (429, 502, 503, 504, 520, 521, 522, 524)
        except Exception as ex:
            last = f"{type(ex).__name__}: {str(ex)[:200]}"
        finally:
            G.release(ok, limited)
        if limited:
            with lock:
                U["rate_limited"] += 1
            errs += 1.0 / 20               # overload retries are cheap but not unbounded (~120 per request)
            time.sleep(2 + 6 * random.random())
        else:
            errs += 1
            time.sleep(min(30, 2 * 2 ** errs))
    raise RuntimeError(last)


def work(r):
    if a.max_minutes and (time.time() - T0) / 60 > a.max_minutes:
        return None
    with lock:
        sk = skipped(r["h"])
    if sk:
        return None
    texts = []
    for s in range(a.samples):
        for att in range(2):
            txt, u = call(r["prompt"])
            with lock:
                U["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                U["completion_tokens"] += int(u.get("completion_tokens") or 0)
                U["reasoning_tokens"] += int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            if txt.strip():
                texts.append(txt); break
        else:
            raise RuntimeError(f"empty content twice (completion_tokens={u.get('completion_tokens')})")
    return r["h"], texts


n_ok = 0
with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
    futs = [ex.submit(work, r) for r in todo]
    for fu in as_completed(futs):
        try:
            res = fu.result()
        except Exception as e:
            with lock:
                U["fail"] += 1
            print("  fail:", str(e)[:160], flush=True)
            continue
        if res is None:
            continue
        h, texts = res
        txt = texts[0]
        rec = {"h": h, "text": txt} if a.raw else {"h": h, "sql": extract_sql(txt)}
        if a.samples > 1:
            rec["sqls"] = [extract_sql(t) for t in texts]
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            U["calls"] += 1
            n_ok += 1
            if n_ok % 100 == 0 or n_ok == len(todo):
                json.dump(U, open(use_path, "w"), indent=1)
                print(f"  {n_ok}/{len(todo)} {(time.time() - T0) / 60:.1f} min  limit={G.lim:.1f}  usage={U}", flush=True)
json.dump(U, open(use_path, "w"), indent=1)
print("done", U, flush=True)
