"""Listwise block reader: one encoder pass reads a whole table (System-One, v2).

Pointwise laya encodes (state + one column question) per column: cost ~ n_cols * |state|.
The block reader asks one choice-type question per (question, table chunk)
    "Which of these columns of table T store information the question asks for?"
with one [MASK] marker per column; each marker logit is trained with sigmoid BCE
(multi-label), initialised from laya_ft_bird.  Cost ~ n_tables * |state|, and columns of
the same table are judged jointly (they compete for the same evidence), which targets the
within-table alias failure mode found by the failure diagnosis.

  train : v2_block_reader.py train            -> <CRISP_MODELS>/laya_block
  score : v2_block_reader.py score --split dev|holdout  -> cache/block_t10_{split}.jsonl
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from jevschema.summary import SummaryConfig, bm25_scores, build_state
from jevschema.backends.laya_backend import stream_load_state_dict

from jevschema.paths import CACHE, MODELS
BASE = os.path.join(MODELS, "laya_ft_bird")
OUT = os.path.join(MODELS, "laya_block")
INS = "Which of the following columns of table `{t}` store information that the question asks for or needs?"
CHUNK = 16


def load_model(path, dev):
    sys.path.insert(0, path)
    import rl_common
    cfg = rl_common.load_cfg(os.path.join(path, "rl_agent_config.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    model = rl_common.build_model(cfg, encoder_dir=os.path.join(path, "encoder"))
    stream_load_state_dict(model, os.path.join(path, "model.safetensors"))
    return rl_common, cfg, tok, model.to(dev)


def table_chunks(ex, tables):
    out = []
    for t in ex.schema.tables:
        if t.name not in tables:
            continue
        cols = [c.name for c in t.columns]
        for i in range(0, len(cols), CHUNK):
            out.append((t.name, cols[i:i + CHUNK]))
    return out


def gate_tables(ex, top_t):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    sc = bm25_scores(q, docs)
    best = {}
    for (t, c), v in zip(pairs, sc):
        best[t] = max(best.get(t, -1e9), v)
    return {t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_t]}


SOFT, ALPHA = {}, 0.7


def make_items(rl, tok, exs, max_len, head_len, all_tables=True, top_t=10):
    items = []
    for e in exs:
        state, tables = build_state(e.question_block(), e.schema, SummaryConfig())
        if all_tables:
            tabs = set(tables)
        elif top_t < 0:
            from jevschema.gating import fk_gate
            tabs = fk_gate(e)
        else:
            tabs = gate_tables(e, top_t)
        gold = e.gold.flat_pairs()
        for tname, cols in table_chunks(e, tabs):
            opts = cols if len(cols) > 1 else cols + ["(no other column)"]   # head needs >= 2 markers
            q = {"t": "choice", "ins": INS.format(t=tname), "crit": {c: "" for c in opts}}
            ids, mk = rl.build_sequence(tok, state, q, max_len, head_len)
            cols = cols[:len(mk)]
            y = [1.0 if (tname, c) in gold else 0.0 for c in cols]
            sd = SOFT.get(e.ex_id)
            if sd:
                y = [ALPHA * sd[(tname, c)] + (1 - ALPHA) * yy if (tname, c) in sd else yy for c, yy in zip(cols, y)]
            items.append({"ids": ids, "markers": mk, "target": y, "qtype": 0, "label": int(max(y) if y else 0),
                          "episode": False, "ep_step": 0, "rec_uid": -1, "cols": [(tname, c) for c in cols]})
    return items


def batches(items, batch_tokens, shuffle, seed=0):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    groups, cur, L = [], [], 0
    for i in order:
        L2 = max(L, len(items[i]["ids"]))
        if cur and L2 * (len(cur) + 1) > batch_tokens:
            groups.append(cur)
            cur, L2 = [], len(items[i]["ids"])
        cur.append(i)
        L = L2
    if cur:
        groups.append(cur)
    if shuffle:
        random.Random(seed).shuffle(groups)
    return groups


def forward_logits(model, rl, items, pad, dev):
    b = rl.collate_items([items], pad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                          b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits.float(), b["marker_mask"].to(dev)


def cmd_train(a):
    from v2data import load_examples
    from jevschema.metrics import auroc
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    rl, cfg, tok, model = load_model(BASE, dev)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    pool = load_examples("pool")
    random.Random(0).shuffle(pool)
    tri = make_items(rl, tok, pool[:3000], a.max_len, a.head_len)
    _soft = dict(SOFT); SOFT.clear()
    evi = make_items(rl, tok, pool[3000:3500], a.max_len, a.head_len)
    SOFT.update(_soft)
    print(f"train items {len(tri)} eval items {len(evi)} pos-rate {np.mean([np.mean(i['target']) for i in tri]):.3f}", flush=True)
    import bitsandbytes as bnb
    from transformers import get_linear_schedule_with_warmup
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01)
    n_groups = len(batches(tri, a.batch_tokens, False))
    sch = get_linear_schedule_with_warmup(opt, max(1, n_groups // 20), n_groups * a.epochs)
    pad = tok.pad_token_id or 0
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        model.train()
        for g in batches(tri, a.batch_tokens, True, seed=ep + 1000 * a.seed):
            its = [tri[i] for i in g]
            logits, mm = forward_logits(model, rl, its, pad, dev)
            tgt = torch.zeros_like(logits)
            for r, it in enumerate(its):
                tgt[r, :len(it["target"])] = torch.tensor(it["target"], device=dev)
            m = mm.float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.clamp(-30, 30), tgt, reduction="none")
            loss = (bce * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  ep{ep} step {step}/{n_groups * a.epochs} loss {loss.item():.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for g in batches(evi, a.batch_tokens, False):
            its = [evi[i] for i in g]
            lg, _ = forward_logits(model, rl, its, pad, dev)
            lg = lg.cpu()
            for r, it in enumerate(its):
                k = len(it["target"])
                ps += torch.sigmoid(lg[r, :k]).tolist()
                ys += it["target"]
    au = float(auroc(np.asarray(ps), np.asarray(ys)))
    print(f"EVAL auroc={au:.4f} n={len(ps)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, os.path.join(OUT, "model.safetensors"))
    import shutil
    for f in ("rl_common.py", "rl_agent_config.json"):
        shutil.copy2(os.path.join(BASE, f), os.path.join(OUT, f))
    for d in ("encoder", "tokenizer"):
        shutil.copytree(os.path.join(BASE, d), os.path.join(OUT, d), dirs_exist_ok=True)
    json.dump({"base": BASE, "ins": INS, "chunk": CHUNK, "max_len": a.max_len, "head_len": a.head_len,
               "epochs": a.epochs, "lr": a.lr, "n_items": len(tri), "eval_auroc": au},
              open(os.path.join(OUT, "train_info.json"), "w"), indent=1)
    print("saved", OUT, flush=True)


def cmd_score(a):
    from v2data import load_examples
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    rl, cfg, tok, model = load_model(OUT, dev)
    model.eval()
    exs = load_examples(a.split)
    sfx = f"_s{a.seed}" if a.seed else ""
    path = os.path.join(CACHE, (f"block_t{a.top_tables}_{a.split}" if a.top_tables > 0 else f"block_fk_{a.split}") + sfx + ".jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
    pad = tok.pad_token_id or 0
    t_all = time.time()
    todo = [x for x in exs if x.ex_id not in done]
    with open(path, "a", encoding="utf-8") as f, torch.no_grad():
        for n, e in enumerate(todo):
            t0 = time.time()
            its = make_items(rl, tok, [e], a.max_len, a.head_len, all_tables=False, top_t=a.top_tables)
            s = {}
            for g in batches(its, a.batch_tokens, False):
                sub = [its[i] for i in g]
                lg, _ = forward_logits(model, rl, sub, pad, dev)
                lg = lg.cpu()
                for r, it in enumerate(sub):
                    for (t, c), v in zip(it["cols"], torch.sigmoid(lg[r, :len(it["cols"])]).tolist()):
                        s[f"{t}|{c}"] = float(v)
            torch.cuda.synchronize()
            f.write(json.dumps({"id": e.ex_id, "s": s, "n_seq": len(its), "enc_tokens": int(sum(len(i["ids"]) for i in its)),
                                "wall_s": round(time.time() - t0, 3)}) + "\n")
            f.flush()
            if (n + 1) % 100 == 0:
                print(f"  {n+1}/{len(todo)} {(time.time()-t_all)/60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--top-tables", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-len", type=int, default=320)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mem-fraction", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--soft", default="")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--base", default="")
    a = ap.parse_args()
    if a.out_dir:
        OUT = a.out_dir
    if a.base:
        BASE = a.base
    if a.soft:
        from v2data import read_scores
        SOFT.update(read_scores(a.soft))
        ALPHA = a.alpha
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.cmd == "train":
        cmd_train(a)
    else:
        cmd_score(a)
