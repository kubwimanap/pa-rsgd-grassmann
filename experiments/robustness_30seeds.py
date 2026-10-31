"""Robustness of the principal interaction claims with 30 seeds.

(a) 2^3 factorial contrasts (9) at every level of Table 4 with 30 common-seed
    replicates (seeds 0-29; seeds 0-7 are those of Table 4), bootstrap 95%
    intervals over seeds (2000 resamples).
(b) Divergence through interaction (eta = 0.3, b = 16, gamma = 0.99) with 30
    seeds and exact (Clopper-Pearson) 95% intervals for the divergence rate.
    A run diverges if an iterate is not finite or its drift exceeds 1e3.

Outputs
  results/tables/contrasts_30seeds.csv
  results/tables/divergence_30seeds.csv
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import beta

import factorial_interaction as F
import replication_and_stability as R
from parsgd.util import bootstrap_ci, parse, pmap, write_csv


def clopper_pearson(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    a = parse(__doc__)
    S = 4 if a.quick else 30
    K = 1500 if a.quick else 4000
    cells = list(itertools.product((0, 1), repeat=3))
    res = pmap(F.job, [dict(level=lv, cell=c, seed=s, K=K) for lv in F.LEVELS for c in cells for s in range(S)],
               a.workers)
    rows = []
    for lv in F.LEVELS:
        for q in ("err", "drift"):
            y = {c: np.array([r[q] for r in res if r["level"] == lv and r["cell"] == c]) for c in cells}
            row = dict(level=lv, quantity="error" if q == "err" else "drift", replicates=S)
            for name, v in F.contrasts(y).items():
                m, lo, hi = bootstrap_ci(v)
                row[f"C_{name}"], row[f"C_{name}_lo"], row[f"C_{name}_hi"] = m, lo, hi
                row[f"C_{name}_excludes_0"] = bool(lo > 0 or hi < 0)
            rows.append(row)
    write_csv("contrasts_30seeds.csv", rows)

    cases = [("noise only (gamma=0, b=16)", "rsgd", 0.0, 16), ("defect only (gamma=0.99, exact gradient)", "rsgd", 0.99, None),
             ("both (gamma=0.99, b=16)", "rsgd", 0.99, 16), ("both, PA-RSGD", "pa", 0.99, 16)]
    cfgs = [dict(case=c, method=m, gamma=g, b=b, seed=s, K=1500) for c, m, g, b in cases for s in range(S)]
    res = pmap(R.div_job, cfgs, a.workers)
    out = []
    for c, *_ in cases:
        rr = [r for r in res if r["case"] == c]
        k = sum(bool(r["div"]) for r in rr)
        lo, hi = clopper_pearson(k, len(rr))
        fin = [r for r in rr if not r["div"]]
        out.append(dict(case=c, diverged=k, runs=len(rr), rate=k / len(rr), ci_lo=lo, ci_hi=hi,
                        error_finite_runs=np.mean([r["err"] for r in fin]) if fin else np.nan))
    write_csv("divergence_30seeds.csv", out)


if __name__ == "__main__":
    main()
