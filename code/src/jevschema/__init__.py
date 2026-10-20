"""Jev-Schema: Risk-Aware Reasoning Routing for Large-Scale Text-to-SQL."""

__version__ = "0.1.0"

import os
import sys

# 让 `python scripts/xxx.py` 直接可用，不需要 pip install -e
_SRC = os.path.dirname(os.path.abspath(__file__))          # .../src/jevschema
_PARENT = os.path.dirname(_SRC)                            # .../src
_ROOT = os.path.dirname(_PARENT)                           # .../jev-schema
for _p in (_PARENT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
