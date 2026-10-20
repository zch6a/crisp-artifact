"""Execution Accuracy（EX）：真跑 SQLite 比对结果集。

论文的端到端指标必须有 EX，否则"省了 token"和"没掉精度"两件事无法同时证明。
这里刻意做得保守：
  * 结果集按**多重集**比对（不做排序敏感比较），与 BIRD 官方口径一致；
  * 数值做相对容差比对（浮点聚合结果 1.0 vs 1 不应判错）；
  * 每条 SQL 有独立超时，超时/报错一律记 fail 而不是跳过 ——
    跳过会让"生成了但跑挂"的样本被当成不存在，EX 会被系统性高估。
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass, field

from .paths import BIRD_DB_DIR
from typing import Any, Dict, List, Optional, Sequence, Tuple

BIRD_SQLITE_DIR = os.environ.get("JEVSCHEMA_BIRD_SQLITE", BIRD_DB_DIR)


class SqlTimeout(Exception):
    pass


def _deadline_handler(deadline):
    import time

    def handler():
        if time.time() > deadline:
            return 1          # 返回非 0 -> 中断查询
        return 0
    return handler


def execute_sql(sqlite_path: str, sql: str, timeout_s: float = 30.0,
                max_rows: int = 100000) -> Tuple[Optional[List[Tuple]], str]:
    """返回 (rows, err)。err 非空表示失败/超时。"""
    if not os.path.exists(sqlite_path):
        return None, f"no_db:{os.path.basename(sqlite_path)}"
    con = None
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, timeout=5.0)
        con.text_factory = lambda b: b.decode("utf-8", "ignore")
        import time
        con.set_progress_handler(_deadline_handler(time.time() + timeout_s), 1000)
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows)
        if cur.fetchone() is not None:
            return None, "too_many_rows"
        return [tuple(r) for r in rows], ""
    except SqlTimeout:
        return None, "timeout"
    except Exception as e:                       # noqa: BLE001
        return None, f"{type(e).__name__}:{str(e)[:80]}"
    finally:
        if con is not None:
            try:
                con.set_progress_handler(None, 0)
                con.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------- 结果比对
def _norm_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int,)):
        return float(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return str(v).strip()


def _norm_rows(rows: Sequence[Tuple]) -> List[Tuple]:
    out = []
    for r in rows:
        out.append(tuple(_norm_cell(x) for x in r))
    return out


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
    return a == b


def result_match(pred: Optional[List[Tuple]], gold: Optional[List[Tuple]]) -> bool:
    """多重集比对（列顺序敏感、行顺序不敏感、数值带容差）。"""
    if pred is None or gold is None:
        return False
    p = _norm_rows(pred)
    g = _norm_rows(gold)
    if len(p) != len(g):
        return False
    if not p and not g:
        return True
    if len(p[0]) != len(g[0]):
        return False
    used = [False] * len(g)
    for row_p in p:
        hit = -1
        for j, row_g in enumerate(g):
            if used[j]:
                continue
            if all(_close(x, y) for x, y in zip(row_p, row_g)):
                hit = j
                break
        if hit < 0:
            return False
        used[hit] = True
    return True


@dataclass
class ExecReport:
    n: int = 0
    correct: int = 0
    gold_fail: int = 0
    pred_fail: int = 0
    errors: Dict[str, int] = field(default_factory=dict)

    @property
    def ex(self) -> float:
        """分母是"gold 能跑通"的样本数 —— gold 自己都跑不通的题不该算我们错。"""
        denom = self.n - self.gold_fail
        return self.correct / denom if denom else 0.0

    @property
    def ex_strict(self) -> float:
        """分母含 gold 失败：诚实上界之外的口径（论文里两个都报，避免挑好看的）。"""
        return self.correct / self.n if self.n else 0.0

    def as_dict(self) -> Dict:
        return {"n": self.n, "correct": self.correct, "ex": round(self.ex, 4),
                "ex_strict": round(self.ex_strict, 4),
                "gold_fail": self.gold_fail, "pred_fail": self.pred_fail,
                "errors": dict(sorted(self.errors.items(), key=lambda kv: -kv[1])[:8])}


def bird_sqlite_path(db_id: str, sqlite_dir: str = BIRD_SQLITE_DIR) -> str:
    return os.path.join(sqlite_dir, db_id, f"{db_id}.sqlite")


def evaluate_execution(pairs: Sequence[Tuple[str, str, str, str]],
                       sqlite_dir: str = BIRD_SQLITE_DIR,
                       timeout_s: float = 30.0,
                       gold_cache: Optional[Dict[str, List[Tuple]]] = None,
                       log_every: int = 0,
                       flags: Optional[list] = None) -> ExecReport:
    """pairs: (db_id, gold_sql, pred_sql, tag)。gold 结果按 db+sql 缓存。

    flags —— 可选的输出数组，逐条记 1（正确）/0（错或预测失败）/None（gold 自身失败，
    应从分母里剔除）。没有它就无法做配对检验：两个系统的 EX 只差 2–3 条时，
    不配对根本看不出这是噪声还是真差异 —— 这正是本项目最需要的那个数。
    """
    rep = ExecReport(n=len(pairs))
    gold_cache = gold_cache if gold_cache is not None else {}
    if flags is not None:
        flags.clear()
    for i, (db_id, gold_sql, pred_sql, _tag) in enumerate(pairs):
        path = bird_sqlite_path(db_id, sqlite_dir)
        gkey = f"{db_id}||{gold_sql}"
        if gkey in gold_cache:
            # a cached gold *failure* must stay a failure (excluded from the denominator); before this fix it was
            # read back as an empty error string, so every system after the first counted it as a wrong prediction
            grows = gold_cache[gkey]
            gerr = "" if grows is not None else gold_cache.get(gkey + "||err", "gold_failed(cached)")
        else:
            grows, gerr = execute_sql(path, gold_sql, timeout_s)
            gold_cache[gkey] = grows if not gerr else None
            if gerr:
                gold_cache[gkey + "||err"] = gerr
        if gerr:
            rep.gold_fail += 1
            rep.errors[f"gold:{gerr[:40]}"] = rep.errors.get(f"gold:{gerr[:40]}", 0) + 1
            if flags is not None:
                flags.append(None)
            continue
        prows, perr = execute_sql(path, pred_sql, timeout_s)
        if perr:
            rep.pred_fail += 1
            rep.errors[f"pred:{perr[:40]}"] = rep.errors.get(f"pred:{perr[:40]}", 0) + 1
            if flags is not None:
                flags.append(0)
            continue
        ok = 1 if result_match(prows, grows) else 0
        if ok:
            rep.correct += 1
        if flags is not None:
            flags.append(ok)
        if log_every and (i + 1) % log_every == 0:
            print(f"    EX {i+1}/{len(pairs)} correct={rep.correct}", flush=True)
    return rep
