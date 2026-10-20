"""Shared data layer for v2 experiments (examples, scorer outputs, oracle depths, costs)."""
from __future__ import annotations

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from jevschema.bird_data import load_bird_dev, load_bird_schemas
from jevschema.bird_train import load_bird_train
from jevschema.certify import oracle_depth, order_from_scores, kept_at
from jevschema.selection import parse_closure
from jevschema.tokens import make_stage_tokens

from jevschema.paths import CACHE, DATA, RESULTS as RES


def read_scores(path):
    store = {}
    if not os.path.exists(path):
        return store
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        store[r["id"]] = {tuple(k.split("|", 1)): float(v) for k, v in r["s"].items()}
    return store


def read_meta(path, keys=("n_seq", "enc_tokens", "wall_s")):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["id"]] = {k: r.get(k) for k in keys}
    return out


def load_examples(split, samples=0):
    if split.startswith("hard"):
        from jevschema.scale_hard import HardScaler
        hs = HardScaler()
        return [hs.merge(e, int(split[4:])) for e in load_examples("dev")]
    if split.startswith("scale"):
        from jevschema.scale import Scaler
        sc = Scaler(load_examples("holdout"))
        return [sc.merge(e, int(split[5:])) for e in load_examples("dev")]
    if split in ("spider2", "spider2raw", "spider2shard"):
        from jevschema.spider2 import load_spider2
        mode = {"spider2": True, "spider2raw": False, "spider2shard": "shard"}[split]
        return load_spider2(os.path.join(DATA, "spider2_lite.json"), canonicalize=mode)
    if split == "spider2sqlite":
        from jevschema.spider2sqlite import load_spider2sqlite
        return load_spider2sqlite()
    if split == "devabbr":
        from jevschema.perturb import perturb_example
        return [perturb_example(e) for e in load_examples("dev")]
    if split == "spider":
        from jevschema.data_loader import load_spider
        return [e for e in load_spider(split="dev", shuffle=False) if e.gold is not None and e.gold.flat_pairs()]
    if split == "dev":
        sch = load_bird_schemas()
        if samples:
            from jevschema.bird_data import enrich_sample_values
            enrich_sample_values(sch, k=samples)
        exs = load_bird_dev(schemas=sch, shuffle=False)
        if os.environ.get("TABREQ") == "1":  # table requirements (v2_tabreq.py)
            from v2_tabreq import augment
            augment(exs)
        return exs
    allx, _ = load_bird_train(limit=None, log_every=0)
    sp = json.load(open(os.path.join(RES, "bird_train_split.json")))
    ids = set(sp["holdout_ids"] if split == "holdout" else sp["pool_ids"])
    return [e for e in allx if e.ex_id in ids]


def _max_norm(v):
    hi = v.max() if len(v) else 0.0
    return v / hi if hi > 0 else np.zeros_like(v)


def _minmax(v):
    if not len(v):
        return v
    lo, hi = v.min(), v.max()
    return (v - lo) / max(1e-9, hi - lo)


def lexical(split):
    if split == "spider":
        return read_scores(os.path.join(CACHE, "bird_spider_bm25.jsonl")), read_scores(os.path.join(CACHE, "lex_embedding_byid.jsonl"))
    if split == "dev":
        return read_scores(os.path.join(CACHE, "bird_bm25.jsonl")), read_scores(os.path.join(CACHE, "bird_embedding.jsonl"))
    return read_scores(os.path.join(CACHE, "train_bm25.jsonl")), read_scores(os.path.join(CACHE, "train_emb.jsonl"))


def fused_scores(ex, bm, em, alpha=0.5):
    """0.5*max-norm(bm25) + 0.5*minmax(emb)  (the project's tier-0 fusion)."""
    pairs = ex.schema.columns()
    b = _max_norm(np.asarray([bm.get(p, 0.0) for p in pairs]))
    e = _minmax(np.asarray([em.get(p, 0.0) for p in pairs])) if em else np.zeros(len(pairs))
    return {p: float((1 - alpha) * x + alpha * y) for p, x, y in zip(pairs, b, e)}


class View:
    """One (split, scorer, closure-config) view: orders, oracle depths, cached kept sizes/tokens."""

    def __init__(self, exs, score_list, cfg_spec):
        self.exs, self.cfg, self.cfg_spec = exs, parse_closure(cfg_spec), cfg_spec
        self.scores = score_list
        self.orders = [order_from_scores(e.schema.columns(), s) for e, s in zip(exs, score_list)]
        self.n_cols = np.asarray([len(e.schema.columns()) for e in exs])
        self.r = np.asarray([oracle_depth(e.schema, o, e.gold, self.cfg) for e, o in zip(exs, self.orders)])
        self._size, self._tok = {}, {}

    def size(self, i, k):
        key = (i, int(k))
        if key not in self._size:
            self._size[key] = len(kept_at(self.exs[i].schema, self.orders[i], k, self.cfg))
        return self._size[key]

    def tokens(self, i, k):
        key = (i, int(k))
        if key not in self._tok:
            e = self.exs[i]
            kept = kept_at(e.schema, self.orders[i], k, self.cfg)
            self._tok[key] = make_stage_tokens(e.schema, e.question, e.evidence or "", kept_columns=kept).gen_prompt_tokens
        return self._tok[key]

    def sorted_scores(self, i):
        s = self.scores[i]
        return np.asarray(sorted((s.get(p, 0.0) for p in self.exs[i].schema.columns()), reverse=True))

    def cum_mass(self, i):
        v = np.maximum(self.sorted_scores(i), 0.0)
        c = np.cumsum(v)
        return c / c[-1] if len(c) and c[-1] > 0 else np.linspace(1.0 / max(1, len(v)), 1.0, len(v))


def full_tokens(exs):
    return np.asarray([make_stage_tokens(e.schema, e.question, e.evidence or "").gen_prompt_tokens for e in exs])
