"""训练 cross-encoder 基线（开题报告 Baseline 4，最主要的竞争对手）。

不训练就没有这个基线：现成的 cross-encoder 没在 (question, column) 相关性上训过，
拿来比是 strawman。这里用 Spider train 的 gold SQL 自动生成监督：
    positive = gold 里出现过的 (table, column)
    negative = 同库里没出现的列（按 1:neg_ratio 采样）

默认用本地已有的 ModernBERT-base（<CRISP_MODELS>/ModernBERT-base），
不重新下载。产出目录：<CRISP_MODELS>/ce_modernbert_base
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.metrics import auroc  # noqa: E402

DEFAULT_ENCODER = os.environ.get("CE_ENCODER", os.path.join(_P.MODELS, "ModernBERT-base"))
DEFAULT_OUT = os.environ.get("CE_OUT", os.path.join(_P.MODELS, "ce_modernbert_base"))


class PairDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len=128):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "label": self.labels[i]}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.float)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        att[i, :len(b["input_ids"])] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": y}


def build_pairs(examples, neg_ratio=4, seed=0, cap=None):
    rng = random.Random(seed)
    texts, labels = [], []
    for ex in examples:
        gold = ex.gold.flat_pairs()
        cols = ex.schema.columns()
        neg_pool = [p for p in cols if p not in gold]
        q = ex.question
        for p in gold:
            texts.append(f"{q} [SEP] table {p[0]}, column {p[1]}")
            labels.append(1)
        k = min(len(neg_pool), neg_ratio * len(gold))
        for p in rng.sample(neg_pool, k):
            texts.append(f"{q} [SEP] table {p[0]}, column {p[1]}")
            labels.append(0)
    if cap and len(texts) > cap:
        idx = rng.sample(range(len(texts)), cap)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--neg-ratio", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--cap-pairs", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    train_ex = load_spider_split(os.path.join(RAW_DIR, "spider_train.parquet"), schemas,
                                 limit=a.n_train, seed=a.seed, shuffle=True)
    eval_ex = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                limit=a.n_eval, seed=a.seed, shuffle=True)
    tr_texts, tr_labels = build_pairs(train_ex, a.neg_ratio, a.seed, a.cap_pairs)
    ev_texts, ev_labels = build_pairs(eval_ex, a.neg_ratio, a.seed + 1)
    print(f"train pairs: {len(tr_texts)} (pos rate {np.mean(tr_labels):.3f}), eval: {len(ev_texts)}")

    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AutoModelForSequenceClassification.from_pretrained(a.encoder, num_labels=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    tr = DataLoader(PairDataset(tr_texts, tr_labels, tok, a.max_len), batch_size=a.batch_size,
                    shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    ev = DataLoader(PairDataset(ev_texts, ev_labels, tok, a.max_len), batch_size=64,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 0))

    steps = int(len(tr) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()

    step = 0
    model.train()
    for ep in range(int(a.epochs) + 1):
        for batch in tr:
            if step >= steps:
                break
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = lossf(out.logits.squeeze(-1), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss={loss.item():.4f}", flush=True)
        if step >= steps:
            break

    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for batch in ev:
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            ps.extend(torch.sigmoid(logits.squeeze(-1)).float().cpu().numpy().tolist())
            ys.extend(batch["labels"].cpu().numpy().tolist())
    auc = auroc(ps, ys)
    print(f"eval AUROC = {auc:.4f} (n={len(ps)})")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump({"encoder": a.encoder, "n_train_examples": len(train_ex),
                   "n_pairs": len(tr_texts), "neg_ratio": a.neg_ratio,
                   "epochs": a.epochs, "lr": a.lr, "eval_auroc": round(float(auc), 4)}, f, indent=2)
    print(f"模型已保存: {a.out}")


if __name__ == "__main__":
    main()
