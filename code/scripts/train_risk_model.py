"""微调 System-One 风险估计器（laya：ModernBERT-large + typed decision head）。

为什么必须微调：零样本下 laya 的列级 AUROC 只有 0.60~0.71，而 BM25 就有 0.80。
Spider 的问题里经常直接出现列名，纯词汇匹配很强，零样本语义判断打不过它 ——
这是必须如实报告的负面结果，也是本方法必须过的一道坎。

微调之后决策模型才具备"Schema Reasoning Risk"的真正语义：
它学的不是语义相似，而是"删掉这列 gold SQL 还能不能写出来"的监督信号。

监督来源：Spider train 的 gold SQL（列级需求集合）。
题型：noul（false / true 两个选项），target 是 true 那一维。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

# transformers 5.x 会把 ModernBERT 的 forward 包进 torch.compile；
# 批内 padded 长度一变就触发一次重编译，实测能把训练拖到 1.7k tok/s（正常 ~30k）。
# 训练里形状本来就在变，直接关掉 dynamo。
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo as _d
    _d.disable()
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jevschema.data_loader import (  # noqa: E402
    RAW_DIR, Example, load_spider_split, load_spider_schemas,
)
from jevschema.metrics import auroc, ece  # noqa: E402
from jevschema.summary import SummaryConfig, build_state  # noqa: E402

DEFAULT_BASE = os.environ.get("LAYA_MODEL_PATH", os.path.join(_P.MODELS, "laya"))
DEFAULT_OUT = os.environ.get("RISK_MODEL_OUT", os.path.join(_P.MODELS, "laya_ft_schema"))

# 训练用的问法（默认用诊断里最好的那条；可用 --template 覆盖）
TEMPLATES = {
    "asks_for": "Does the question ask for information that is stored in the column `{t}.{c}`?",
    "needed": "Answering the question requires using the column `{t}.{c}` in the SQL query.",
    "risk": ("If the column `{t}.{c}` were removed from this database schema, would writing the "
             "correct SQL to answer the question fail or become substantially harder?"),
    "risk_deleted": ("Suppose the column `{t}.{c}` is deleted from the database. It would then be "
                     "impossible to write a correct SQL query that answers the question."),
}


def build_items(rl, tok, examples, template, max_len, head_max_len, max_neg, seed=0):
    """把每个 (example, column) 编成一条训练序列。"""
    rng = random.Random(seed)
    items = []
    for ex in examples:
        state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
        gold = ex.gold.flat_pairs()
        cols = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
        pos = [p for p in cols if p in gold]
        neg = [p for p in cols if p not in gold]
        if max_neg and len(neg) > max_neg:
            neg = rng.sample(neg, max_neg)
        for (t, c), lab in [(p, 1) for p in pos] + [(p, 0) for p in neg]:
            qq = {"t": "noul", "ins": template.format(t=t, c=c), "crit": {}}
            ids, markers = rl.build_sequence(tok, state, qq, max_len, head_max_len)
            if len(markers) < 2:
                continue
            items.append({"ids": ids, "markers": markers, "target": [1.0 - lab, float(lab)],
                          "qtype": 2, "label": lab, "episode": False, "ep_step": 0,
                          "rec_uid": -1})
    return items


def _merge_distractors(examples, schemas, target_cols: int, seed: int = 0) -> list:
    """给每个训练样本并入其它库的干扰表，直到列数达到 target_cols，gold 保持不变。

    两个目的：
      1. BIRD-train 的标注 DDL 平均只有 49 列，而 BIRD-Dev 是 76 列、工业库上千列。
         只在小库上训练，模型从没见过"99% 的列都是无关列"的分布；
      2. 干扰表里经常有与 gold 同名的列，逼模型学会靠表名/上下文消歧，
         而不是靠"这个名字看起来眼熟"。
    """
    from jevschema.schema_graph import Column, SchemaGraph, Table

    rng = random.Random(seed)
    pool_all = [g for g in schemas.values()]
    out = []
    for ex in examples:
        base = ex.schema
        if base.n_columns() >= target_cols:
            out.append(ex)
            continue
        tables = [Table(t.name, [Column(c.name, c.dtype, c.is_pk, list(c.sample_values))
                                 for c in t.columns]) for t in base.tables]
        fks = list(base.fks)
        n = base.n_columns()
        pool = [g for g in pool_all if g.db_id != base.db_id]
        rng.shuffle(pool)
        for g in pool:
            if n >= target_cols:
                break
            pfx = g.db_id + "_"
            for t in g.tables:
                tables.append(Table(pfx + t.name,
                                    [Column(c.name, c.dtype, c.is_pk, []) for c in t.columns]))
                n += len(t.columns)
        new = SchemaGraph(db_id=f"{base.db_id}_m{target_cols}", tables=tables, fks=fks)
        ex2 = Example(ex_id=ex.ex_id, db_id=new.db_id, question=ex.question, sql=ex.sql,
                      schema=new, evidence=ex.evidence, difficulty=ex.difficulty)
        ex2.gold = ex.gold          # 基表不改名，gold pair 仍然有效
        out.append(ex2)
    return out


def run_batches(model, rl, tok, items, pad_id, dev, batch_tokens, optimizer=None,
                pos_weight=1.0, amp="bf16", log_every_=100, scheduler=None,
                max_steps=0, eval_items=None, eval_every=0):
    model.train() if optimizer else model.eval()
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    i, step, tot_loss, t0 = 0, 0, 0.0, time.time()
    done_tok, batch_seq, recent, n_oom = 0, [], [], 0
    probs_out, labels_out, margins_out = [], [], []
    grad_ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with grad_ctx:
        while i < len(order):
            j, L = i, 0
            while j < len(order):
                L = max(L, len(items[order[j]]["ids"]))
                if L * (j - i + 1) > batch_tokens:
                    break
                j += 1
            j = max(j, i + 1)
            sel = [items[t] for t in order[i:j]]
            b = rl.collate_items([sel], pad_id)
            tgt = torch.zeros_like(b["target"])
            for r, it in enumerate(sel):
                tgt[r, :len(it["target"])] = torch.tensor(it["target"])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            tgt = tgt.to(dev)
            try:
                # 必须用 bf16：官方 rl_agent_config.json 的 amp_dtype 就是 bf16，
                # ModernBERT 在 fp16 下中间激活会溢出（实测整个训练塌成常数预测）。
                with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=bool(amp) and dev.type == "cuda"):
                    logits, _act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                         b["marker_mask"], b["qtype"])
            except torch.OutOfMemoryError:
                # 显存被本机其它任务随时抢占（同一张卡上还有别人的 finetune 在跑），
                # 训练不该因此整体崩掉：跳过这一批并释放缓存即可。
                n_oom += 1
                del b
                torch.cuda.empty_cache()
                i = j
                continue
            except Exception as e:                   # noqa: BLE001
                # 共享 GPU 上的 CUDA 错误往往不是 OOM（实测是 cudaErrorUnknown），
                # 但同样只是"这一批没跑成"。不捕获的话整个训练会在第 1 步就死掉。
                if "CUDA" in str(e) or "Accelerator" in type(e).__name__:
                    n_oom += 1
                    del b
                    torch.cuda.empty_cache()
                    i = j
                    continue
                raise
            logp = torch.log_softmax(logits.float(), -1)
            w = torch.where(tgt > 0.5, torch.full_like(tgt, pos_weight), torch.ones_like(tgt))
            loss = -(w * tgt * logp * b["marker_mask"]).sum() / b["marker_mask"].sum().clamp(min=1)
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()      # 必须每个 batch 步进：按 epoch 步进会让 LR
                                          # 长期停在 warmup 起点，前几千步等于没学
                optimizer.zero_grad(set_to_none=True)
            tot_loss += float(loss.item())
            recent.append(float(loss.item()))
            if len(recent) > 100:
                recent.pop(0)
            p = torch.softmax(logits.float(), -1)[:, 1]
            probs_out.extend(p.detach().cpu().numpy().tolist())
            # 两个 marker 的 logit 差：用来在训练后重新拟合 temperature
            margins_out.extend((logits.float()[:, 1] - logits.float()[:, 0])
                               .detach().cpu().numpy().tolist())
            labels_out.extend([it["label"] for it in sel])
            step += 1
            done_tok += int(b["attention_mask"].sum())
            batch_seq.append(j - i)
            i = j
            if optimizer and step % log_every_ == 0:
                el = time.time() - t0
                tps = done_tok / max(el, 1e-9)
                avg_b = float(np.mean(batch_seq[-100:]))
                n_batches = len(order) / max(1.0, avg_b)
                print(f"  step {step} loss={tot_loss/step:.4f} recent={np.mean(recent):.4f} "
                      f"{tps/1000:.1f}k tok/s "
                      f"seq/batch={avg_b:.1f} eta {(n_batches-step)*el/step/60:.1f}m", flush=True)
            if (eval_items and eval_every and optimizer and step % eval_every == 0):
                was = model.training
                model.eval()
                _l2, pe, ye, _me = run_batches(model, rl, tok, eval_items, pad_id, dev,
                                          batch_tokens, None, pos_weight, amp=amp)
                model.train() if was else model.eval()
                print(f"    [mid-eval @{step}] auroc={auroc(pe, ye):.4f} "
                      f"pos_mean={pe[ye == 1].mean():.4f} neg_mean={pe[ye == 0].mean():.4f}",
                      flush=True)
            if max_steps and step >= max_steps:
                print(f"  (达到 max_steps={max_steps}，提前停止)", flush=True)
                break
    return (tot_loss / max(1, step), np.asarray(probs_out), np.asarray(labels_out),
            np.asarray(margins_out))


def fit_temperature(margin, y, cfg_temperature=None):
    """在两个 marker 的 logit 差上重拟合 temperature（最小化 NLL，网格 + 局部细化）。

    为什么单独做这一步：基座的温度是按微调前的 logit 尺度拟合的，微调后模型自信得多，
    沿用旧温度会把概率整体压平 —— 路由阈值 τ 和 ECE 都会失真。
    """
    m = np.asarray(margin, dtype=float)
    yy = np.asarray(y, dtype=float)
    if m.size == 0 or yy.sum() == 0 or yy.sum() == yy.size:
        return 1.0, float("nan"), float("nan")
    T0 = float(cfg_temperature[2]) if cfg_temperature else 1.0

    def nll(T: float) -> float:
        z = np.clip(m / max(T, 1e-6), -60, 60)
        # log sigmoid(z) 与 log(1-sigmoid(z)) 的稳定写法
        return float(np.mean(np.logaddexp(0.0, -z) * yy + np.logaddexp(0.0, z) * (1 - yy)))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best = float(grid[int(np.argmin([nll(t) for t in grid]))])
    lo, hi = best / 1.5, best * 1.5
    for _ in range(40):                       # 三分查找细化
        a_, b_ = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(a_) < nll(b_):
            hi = b_
        else:
            lo = a_
    T = float((lo + hi) / 2)
    p_base = 1.0 / (1.0 + np.exp(-np.clip(m / max(T0, 1e-6), -60, 60)))
    p_new = 1.0 / (1.0 + np.exp(-np.clip(m / max(T, 1e-6), -60, 60)))
    return T, float(ece(p_base, yy)[0]), float(ece(p_new, yy)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--template", default="asks_for", choices=list(TEMPLATES))
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--max-neg", type=int, default=40)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw", "adamw8bit"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--encoder", default="", help="覆盖基座 encoder 目录（如 ModernBERT-base）")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--overfit", type=int, default=0,
                    help=">0 时只用前 N 条做训练+验证，用于自检（必须能把 AUROC 推到 ~1）")
    ap.add_argument("--eval-every", type=int, default=0, help="训练中每 N 步在小验证集上评估一次")
    ap.add_argument("--max-steps", type=int, default=0, help="最多训练多少步（0=不限）")
    ap.add_argument("--eval-source", default="train", choices=["train", "val"])
    ap.add_argument("--train-data", default="spider", choices=["spider", "bird", "both"],
                    help="bird = BIRD train 域内数据（arcwise 标注），两者混训用 both")
    ap.add_argument("--n-train-bird", type=int, default=0, help="0 = 用全部 BIRD train")
    ap.add_argument("--max-len", type=int, default=0, help="覆盖 cfg.max_len（BIRD 需要 >=1024）")
    ap.add_argument("--merge-target-cols", type=int, default=0,
                    help=">0 时给每个训练样本并入干扰表直到列数达标，gold 不变。"
                         "用来教会模型拒绝无关列，并匹配 BIRD-Dev 更大的 schema。")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="冻结 encoder，只训 decision head（省显存、防大模型被高 LR 打坏）")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help=">0 时给本进程设显存硬上限（占总额的比例）。本机 GPU 常与他人训练共享，"
                         "没有上限的话我们一旦冲高会把对方挤到 OOM。")
    a = ap.parse_args()
    amp = None if a.amp_dtype == "none" else a.amp_dtype

    if a.base not in sys.path:
        sys.path.insert(0, a.base)
    import rl_common  # type: ignore
    from safetensors.torch import save_file
    from jevschema.backends.laya_backend import stream_load_state_dict
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    enc_dir = a.encoder or os.path.join(a.base, "encoder")
    cfg = rl_common.load_cfg(os.path.join(a.base, "rl_agent_config.json"))
    cfg["encoder"] = enc_dir
    tok = AutoTokenizer.from_pretrained(os.path.join(a.base, "tokenizer"))
    model = rl_common.build_model(cfg, encoder_dir=enc_dir)
    base_path = os.path.join(a.base, "model.safetensors")
    if os.path.exists(base_path) and not a.encoder:
        # 本机提交内存长期贴着上限，mmap 会被 Windows 拒（OSError 1455），
        # 而这不是我们能根治的（改页面文件要管理员）。所以：重试 + 让出内存后再试。
        import gc

        n = 0
        for attempt in range(6):
            try:
                n = stream_load_state_dict(model, base_path)
                break
            except OSError as e:
                print(f"[retry {attempt+1}] 权重 mmap 失败：{e}")
                gc.collect()
                time.sleep(20)
        if n == 0:
            raise RuntimeError("基座权重加载失败（页面文件不足）；请关闭其它占内存的进程后重试")
        print(f"已流式加载基座权重 {n} 个张量: {base_path}")
    elif a.encoder:
        print(f"换用 encoder={enc_dir}：decision head 随机初始化（基座 head 维度不匹配，丢弃）")
    if a.freeze_encoder:
        for prm in model.encoder.parameters():
            prm.requires_grad_(False)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"encoder 已冻结，可训练参数 {n_tr/1e6:.1f}M")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mem_fraction > 0 and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, device=dev)
        print(f"显存硬上限: {a.mem_fraction:.2f} × 总额")
    if not a.no_grad_ckpt:
        # ModernBERT-large 全量微调 + 16k token/批 会直接 OOM（实测 13.2 GiB）。
        # 激活重算是这里唯一能同时保住 batch 和全量微调的办法。
        try:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.head_checkpointing = True
            print("gradient checkpointing: on")
        except Exception as e:
            print(f"[warn] 未能开启 gradient checkpointing: {e}")
    model.to(dev)
    pad_id = tok.pad_token_id or 0
    max_len = int(a.max_len or cfg.get("max_len", 512))
    cfg["max_len"] = max_len          # 必须写回：保存的 cfg 决定推理时的截断长度
    head_max_len = int(cfg.get("head_max_len", 192))

    if a.eval_only:
        stream_load_state_dict(model, os.path.join(a.out, "model.safetensors"))
        model.to(dev)

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))

    def _spider_pool(n: int) -> list:
        return load_spider_split(os.path.join(RAW_DIR, "spider_train.parquet"), schemas,
                                 limit=n, seed=a.seed, shuffle=True)

    def _bird_pool(n: int) -> list:
        from jevschema.bird_train import load_bird_train
        exs, st = load_bird_train(limit=n or None)
        print(f"BIRD train: {len(exs)} 例 / {st['n_dbs']} 库 / 平均 {st['avg_columns']:.1f} 列 "
              f"/ 平均 gold {st['avg_gold']:.2f}（no_gold {st['no_gold']}）")
        # 打乱后切片：文件按库聚簇，取前缀会让验证集全是同一批库
        random.Random(a.seed).shuffle(exs)
        return exs

    # 验证集必须来自**与训练集同分布但不相交**的一段。
    # 用 spider_val 当验证集等于在测试集上调参（Spider 的 dev 全程不许碰）。
    if a.train_data == "bird":
        pool = _bird_pool(a.n_train_bird or (a.n_train + a.n_eval))
        train_ex, eval_ex = pool[:a.n_train], pool[a.n_train:a.n_train + a.n_eval]
        print(f"数据源: BIRD train | train {len(train_ex)} / eval {len(eval_ex)}（同一池内不相交切片）")
    elif a.train_data == "both":
        sp = _spider_pool(a.n_train + a.n_eval)
        bd = _bird_pool(a.n_train_bird or (a.n_train + a.n_eval))
        pool = sp + bd
        random.Random(a.seed).shuffle(pool)
        train_ex, eval_ex = pool[:a.n_train], pool[a.n_train:a.n_train + a.n_eval]
        print(f"数据源: Spider+BIRD 混训 | train {len(train_ex)} / eval {len(eval_ex)}")
    else:
        if a.eval_source == "train":
            pool = _spider_pool(a.n_train + a.n_eval)
            train_ex, eval_ex = pool[:a.n_train], pool[a.n_train:]
            print("数据源: Spider train | 验证集=train holdout（不相交，dev 全程未使用）")
        else:
            train_ex = _spider_pool(a.n_train)
            eval_ex = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                        limit=a.n_eval, seed=a.seed + 1, shuffle=True)
            print("数据源: Spider train | 验证集=spider_val（仅最终对照，不作调参依据）")

    if a.merge_target_cols > 0:
        train_ex = _merge_distractors(train_ex, schemas, a.merge_target_cols, a.seed)
        eval_ex = _merge_distractors(eval_ex, schemas, a.merge_target_cols, a.seed + 1)
    tmpl = TEMPLATES[a.template]
    tr_items = build_items(rl_common, tok, train_ex, tmpl, max_len, head_max_len, a.max_neg, a.seed)
    ev_items = build_items(rl_common, tok, eval_ex, tmpl, max_len, head_max_len, a.max_neg, a.seed + 1)
    if a.overfit:
        tr_items = tr_items[:a.overfit]
        ev_items = tr_items[:a.overfit]
        print(f"[overfit 自检] 只用 {len(tr_items)} 条，训练后 train/eval AUROC 都应接近 1")
    print(f"train items: {len(tr_items)} (pos {np.mean([i['label'] for i in tr_items]):.3f}), "
          f"eval items: {len(ev_items)}")

    if not a.eval_only:
        if a.optim == "adamw8bit":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01)
            print("optimizer: AdamW8bit（fp32 参数 + 8bit 状态，省 ~4.8GB）")
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        # 必须用真实序列长度估算步数：写死 512 会让 max_len=1024 时的 scheduler
        # 周期短一半，LR 提前衰减到 0，后半程等于没在学。
        n_batches = max(1, int(len(tr_items) * max_len / a.batch_tokens))
        sch = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * n_batches)),
                                              max(1, int(n_batches * a.epochs)))
        done = 0
        ep = 0
        while done < a.epochs:
            ep += 1
            print(f"epoch {ep}: {len(tr_items)} items")
            loss, p, y, _m = run_batches(model, rl_common, tok, tr_items, pad_id, dev,
                                     a.batch_tokens, opt, a.pos_weight, amp=amp,
                                     log_every_=a.log_every, scheduler=sch,
                                     max_steps=a.max_steps,
                                     eval_items=(ev_items[:1500] if a.eval_every else None),
                                     eval_every=a.eval_every)
            print(f"  train loss={loss:.4f} auroc={auroc(p, y):.4f}", flush=True)
            done += 1
            if a.max_steps and a.max_steps * ep >= len(tr_items):
                break

    model.eval()
    _l, p, y, margin = run_batches(model, rl_common, tok, ev_items, pad_id, dev, a.batch_tokens,
                                   amp=amp)
    print(f"EVAL auroc={auroc(p, y):.4f} pos_mean={p[y==1].mean():.4f} "
          f"neg_mean={p[y==0].mean():.4f} n={len(p)}", flush=True)

    # ---- 重新拟合 noul 的 temperature ----
    # 基座的温度是给微调前的 logit 尺度用的；微调后模型自信得多，沿用旧温度会把
    # 概率整体压平，路由阈值和 ECE 都会失真。这里在验证集上最小化 NLL 重拟合。
    T, ece_before, ece_after = fit_temperature(margin, y, cfg_temperature=cfg.get("temperature"))
    print(f"temperature: noul 基座={cfg.get('temperature', [1,1,1])[2]:.3f} -> 重拟合={T:.3f} "
          f"(ECE {ece_before:.4f} -> {ece_after:.4f})", flush=True)
    if not a.eval_only:
        cfg["temperature"] = [cfg.get("temperature", [1.0, 1.0, 1.0])[0],
                              cfg.get("temperature", [1.0, 1.0, 1.0])[1], float(T)]
        cfg.setdefault("temperature_by_options", {})["noul:2"] = float(T)

    if not a.eval_only:
        os.makedirs(a.out, exist_ok=True)
        save_file(model.state_dict(), os.path.join(a.out, "model.safetensors"))
        tok.save_pretrained(os.path.join(a.out, "tokenizer"))
        import shutil
        shutil.copytree(os.path.join(a.base, "encoder"), os.path.join(a.out, "encoder"),
                        dirs_exist_ok=True)
        for f in ("rl_common.py", "rl_agent_api.py", "rl_agent_config.json", "email_utils.py"):
            src = os.path.join(a.base, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(a.out, f))
        with open(os.path.join(a.out, "train_info.json"), "w", encoding="utf-8") as f:
            json.dump({"base": a.base, "encoder": enc_dir, "template": a.template,
                       "train_data": a.train_data, "max_len": max_len,
                       "merge_target_cols": a.merge_target_cols,
                       "n_train_examples": a.n_train, "n_items": len(tr_items),
                       "epochs": a.epochs, "lr": a.lr, "batch_tokens": a.batch_tokens,
                       "pos_weight": a.pos_weight, "eval_auroc": round(float(auroc(p, y)), 4)},
                      f, indent=2)
        with open(os.path.join(a.out, "rl_agent_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"模型已保存: {a.out}")


if __name__ == "__main__":
    main()
