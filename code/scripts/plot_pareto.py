"""画图：recall-filter 曲线、scaling 曲线、成本-延迟散点、reliability diagram。

输入都是 results/ 下的 JSON，不重跑实验。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figs")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_recall_filter(curves_path, pilot_path, out):
    d = _load(curves_path)
    plt.figure(figsize=(6.5, 4.5))
    for name, curve in d["curves"].items():
        xs = [float(k) for k in curve]
        ys = [curve[k] for k in curve]
        plt.plot(xs, ys, marker="o", ms=3, label=f"{name} (AUROC {d['auroc'].get(name)})")
    if pilot_path and os.path.exists(pilot_path):
        p = _load(pilot_path)
        m = p["metrics"]
        plt.scatter([m["filter_rate"]], [m["recall"]], marker="*", s=220, color="crimson",
                    label=f"Jev-Schema tau={p['config']['tau_lo']}/{p['config']['tau_hi']}", zorder=5)
    plt.xlabel("Filter rate (fraction of columns removed)")
    plt.ylabel("Schema recall")
    plt.title("Recall vs filter rate (Spider dev)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved", out)


def plot_scaling(path, out):
    d = _load(path)
    rows = d["rows"]
    xs = [r["actual_columns"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(xs, [r["recall"] for r in rows], "o-", color="tab:blue", label="recall")
    ax1.plot(xs, [r["filter_rate"] for r in rows], "s--", color="tab:green", label="filter rate")
    ax1.set_xscale("log")
    ax1.set_xlabel("schema size (columns, log)")
    ax1.set_ylabel("recall / filter rate")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["scoring_ms_per_example"] or 0 for r in rows], "x--", color="tab:red",
             label="ms/example")
    ax2.set_ylabel("scoring ms per example")
    fig.legend(loc="lower left", fontsize=8)
    plt.title("Schema scaling")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved", out)


def plot_reliability(cal_path, out):
    d = _load(cal_path)
    rel = d.get("reliability", [])
    xs = [0.5 * (r["lo"] + r["hi"]) for r in rel if r["n"]]
    ys = [r["acc"] for r in rel if r["n"]]
    ns = [r["n"] for r in rel if r["n"]]
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    plt.plot(xs, ys, "o-", label="model")
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("predicted risk")
    plt.ylabel("empirical P(column is required)")
    plt.title(f"Reliability (ECE={d.get('ece_uniform_eval')}, AUROC={d.get('auroc_eval')})")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=os.path.join(RESULTS, "baselines_n200.json"))
    ap.add_argument("--pilot", default="")
    ap.add_argument("--scaling", default="")
    ap.add_argument("--calibration", default="")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    if os.path.exists(a.curves):
        plot_recall_filter(a.curves, a.pilot, os.path.join(FIGDIR, "recall_filter.png"))
    if a.scaling and os.path.exists(a.scaling):
        plot_scaling(a.scaling, os.path.join(FIGDIR, "scaling.png"))
    if a.calibration and os.path.exists(a.calibration):
        plot_reliability(a.calibration, os.path.join(FIGDIR, "reliability.png"))


if __name__ == "__main__":
    main()
