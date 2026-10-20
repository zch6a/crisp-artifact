"""Label efficiency of the certified fallback (Table 4 / Section 7.4).

    python reproduce/label_efficiency.py --r2 results/cert232_r2_k16.json --eps 0.05

For each calibration size, draw random calibration/test splits and report the mean and
5th percentile of realized coverage and the mean certified size.
"""
import argparse, json, math, os
import numpy as np


def qhat(vals, eps):
    v = np.sort(vals); j = math.ceil((len(v) + 1) * (1 - eps))
    return int(v[j - 1]) if j <= len(v) else 10 ** 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sizes", default="25,50,100,200,400,767")
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args()
    d = json.load(open(a.r2, encoding="utf-8"))
    r2 = np.asarray(d["r2"])
    size = {int(k): np.asarray(v) for k, v in d["size"].items()}
    n = len(r2); rng = np.random.default_rng(0)
    for ncal in [int(x) for x in a.sizes.split(",")]:
        cov, sz = [], []
        for _ in range(a.reps):
            p = rng.permutation(n)
            lam = qhat(r2[p[:ncal]], a.eps)
            cov.append(np.mean(r2[p[ncal:]] <= lam))
            sz.append(np.mean(size[lam][p[ncal:]]) if lam in size else np.nan)
        print(json.dumps({"n_cal": ncal, "coverage_mean": float(np.mean(cov)),
                          "coverage_p05": float(np.percentile(cov, 5)),
                          "size_mean": float(np.nanmean(sz))}))


if __name__ == "__main__":
    main()
