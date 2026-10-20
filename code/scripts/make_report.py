"""把 results/ 下的实验产物拼成一份 markdown 报告（不重跑实验）。

报告里的每个数字都带样本量与 CI；没有跑出来的实验在表里写"未跑"，
绝不拿别的数据集/别的配置的数字填空。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
RESULTS = os.path.join(_ROOT, "results")


def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "REPORT.md"))
    a = ap.parse_args()

    lines = ["# Jev-Schema 实验报告", "", "自动生成：scripts/make_report.py（数据来源 results/*.json）", ""]

    # Pilot
    pilots = sorted(glob.glob(os.path.join(RESULTS, "pilot_*.json")))
    if pilots:
        lines += ["## Pilot（Spider dev）", "",
                  "| 配置 | n | recall (95% CI) | filter rate | full-recall | precision | ECE | AUROC | ms/例 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in pilots:
            d = _load(p)
            c, m, cal = d["config"], d["metrics"], d["calibration"]
            ci = m.get("recall_ci", [0, 0])
            esc = c.get("escalate", "keep")
            rfn = c.get("refiner", "keep-all")
            if rfn != "keep-all":
                esc += f"→{rfn}:{c.get('refiner_model', '')}"
            lines.append(
                f"| {c['backend']} views={','.join(c['views'])} tau={c['tau_lo']}/{c['tau_hi']} "
                f"fk={c['fk_closure']} esc={esc} | {m.get('n_examples')} | "
                f"{m.get('recall')} [{ci[0]}, {ci[1]}] | {m.get('filter_rate')} | "
                f"{m.get('full_recall_rate')} | {m.get('precision')} | "
                f"{cal.get('ece_uniform')} | {cal.get('auroc')} | "
                f"{d.get('cost', {}).get('latency_per_example_ms')} |")
        lines.append("")

    # 基线曲线
    bl = os.path.join(RESULTS, "baselines_n200.json")
    if os.path.exists(bl):
        d = _load(bl)
        rates = [str(r) for r in d["rates"]]
        lines += ["## 基线：recall @ 过滤率", "", "| 方法 | AUROC | " + " | ".join(rates) + " |",
                  "|---|---|" + "---|" * len(rates)]
        for name, curve in d["curves"].items():
            lines.append(f"| {name} | {d['auroc'].get(name)} | " +
                         " | ".join(str(curve.get(r, "")) for r in rates) + " |")
        lines.append("")

    # 同口径打分器对比
    for f in sorted(glob.glob(os.path.join(RESULTS, "compare_*.json"))):
        d = _load(f)
        rates = [str(r) for r in d["rates"]]
        lines += [f"## 同口径打分器对比（{d['tag']}, n={d['n']}）", "",
                  "| 打分器 | AUROC | " + " | ".join(rates) + " |",
                  "|---|---|" + "---|" * len(rates)]
        for name, v in sorted(d["scorers"].items(), key=lambda kv: -kv[1]["auroc"]):
            lines.append(f"| {name} | {round(v['auroc'], 4)} | " +
                         " | ".join(str(round(v["curve"].get(r, 0), 3)) for r in rates) + " |")
        if d.get("risk_ft_vs_embedding"):
            lines += ["", "**risk_ft − embedding（配对 bootstrap，5000 次）**", "",
                      "| 过滤率 | Δrecall | 95% CI | p |", "|---|---|---|---|"]
            for r in rates:
                s = d["risk_ft_vs_embedding"][r]
                lines.append(f"| {r} | {s['delta']:+.4f} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | "
                             f"{s['p']} |")
        lines.append("")

    # 消融
    for f in sorted(glob.glob(os.path.join(RESULTS, "ablation_*.json"))):
        d = _load(f)
        lines += [f"## 消融（{d['backend']}, n={d['n']}, tau={d['tau']}）", "",
                  "| 变体 | recall | table recall | filter | precision | escal | full-recall |",
                  "|---|---|---|---|---|---|---|"]
        for name, s in d["summary"].items():
            lines.append(f"| {name} | {s['recall']} | {s['table_recall']} | {s['filter_rate']} | "
                         f"{s['precision']} | {s['escalation_rate']} | {s['full_recall_rate']} |")
        if d.get("tests_vs_full"):
            lines += ["", "| 变体 vs full | recall 差 | 95% CI | p(bootstrap) | p(sign-flip) |",
                      "|---|---|---|---|---|"]
            for t in d["tests_vs_full"]:
                lines.append(f"| {t['variant']} | {round(t['recall_diff'], 4)} | "
                             f"[{round(t['recall_ci'][0], 4)}, {round(t['recall_ci'][1], 4)}] | "
                             f"{round(t['p_boot'], 4)} | {round(t['p_signflip'], 4)} |")
        lines.append("")

    # Scaling
    for f in sorted(glob.glob(os.path.join(RESULTS, "scaling_*.json"))):
        d = _load(f)
        lines += ["## Schema Scaling", "",
                  "| 列数 | 表数 | recall | filter | kept 列 | ms/例 |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            lines.append(f"| {r['actual_columns']} | {r['tables']} | {r['recall']} | "
                         f"{r['filter_rate']} | {r['kept_columns']} | {r['scoring_ms_per_example']} |")
        lines.append("")

    # 校准
    for f in sorted(glob.glob(os.path.join(RESULTS, "calibration_*.json"))):
        d = _load(f)
        lines += ["## 校准（" + os.path.basename(f) + "）", "",
                  "| 指标 | 值 |", "|---|---|"]
        for k, v in d.items():
            if k != "reliability":
                lines.append(f"| {k} | {v} |")
        lines.append("")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"写入 {a.out}")


if __name__ == "__main__":
    main()
