"""Large spiked problem (n = 10^4, p = 10) under N+S+G (t=8, b=32, gamma=0.95).

The covariance is never formed.  The tuned constant step is chosen from the
grid of Table 6 on separate tuning seeds.

Outputs
  results/tables/large_spiked.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, pa_rsgd, random_start, rsgd, spiked
from parsgd.util import mean_ci, parse, pmap, write_csv

P = spiked(0)
R = Retraction("surrogate", 0.95)
T, B = 8, 32


def job(c):
    rng = np.random.default_rng(c["seed"])
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(T, c.get("mode", "RN"), rng)
    if c["method"] == "pa":
        tr = pa_rsgd(P, X0, c["K"], B, R, ar, rng, rec=50)
    else:
        tr = rsgd(P, X0, c["K"], c["eta"], B, R, ar, rng, rec=50, decay=c.get("decay"))
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"), div=tr.diverged)


def main():
    a = parse(__doc__)
    K = 300 if a.quick else 2000
    grid = np.geomspace(0.003, 0.3, 4 if a.quick else 10)
    res = pmap(job, [dict(method="rsgd", eta=e, seed=100 + i, K=K) for e in grid for i in range(1 if a.quick else 2)],
               a.workers)
    tuned = min(grid, key=lambda e: np.mean([r["err"] for r in res if r["eta"] == e]))
    test = range(200, 202) if a.quick else range(200, 206)
    methods = {
        "RSGD, eta=0.02": dict(method="rsgd", eta=0.02),
        "RSGD, decaying": dict(method="rsgd", eta=0.02, decay=200),
        f"RSGD, tuned eta={tuned:.3g}": dict(method="rsgd", eta=tuned),
        f"RSGD, tuned eta={tuned:.3g}, SR": dict(method="rsgd", eta=tuned, mode="SR"),
        "PA-RSGD": dict(method="pa"),
    }
    cfgs = [dict(m, label=k, seed=i, K=K) for k, m in methods.items() for i in test]
    res = pmap(job, cfgs, a.workers)
    rows = []
    for k in methods:
        rr = [r for r in res if r["label"] == k]
        m, h = mean_ci([r["err"] for r in rr])
        d, _ = mean_ci([r["drift"] for r in rr])
        rows.append(dict(method=k, error_mean=m, error_ci=h, drift_mean=d,
                         diverged=sum(r["div"] for r in rr), seeds=len(rr), max_error=np.sqrt(P.p)))
    write_csv("large_spiked.csv", rows)


if __name__ == "__main__":
    main()
