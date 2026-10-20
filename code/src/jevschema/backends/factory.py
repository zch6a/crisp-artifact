"""后端工厂：一个字符串拿到后端，实验脚本不 import 具体类。"""

from __future__ import annotations

import os
from typing import Optional

from .base import CachedBackend, DecisionBackend, MockBackend, RandomBackend
from ..paths import CACHE

CACHE_DIR = os.environ.get("JEVSCHEMA_CACHE", CACHE)


def build_backend(name: str = "thisthat", cached: bool = True, dry_run: bool = True, **kw) -> DecisionBackend:
    name = (name or "mock").lower()
    if name in ("mock", "none"):
        be: DecisionBackend = MockBackend()
    elif name == "random":
        be = RandomBackend(kw.pop("seed", 0))
    elif name in ("thisthat", "this-that", "tt"):
        from .thisthat_backend import ThisThatBackend
        be = ThisThatBackend(**kw)
    elif name == "laya":
        from .laya_backend import LayaBackend
        be = LayaBackend(**kw)
    elif name == "jev":
        from .jev_backend import JevBackend
        be = JevBackend(dry_run=dry_run, **kw)
    else:
        raise ValueError(f"未知后端: {name}")
    if cached and name != "mock":
        be = CachedBackend(be, os.path.join(CACHE_DIR, f"{be.name}.jsonl"))
    return be


def close_backend(be: Optional[DecisionBackend]) -> None:
    if isinstance(be, CachedBackend):
        be.flush()
