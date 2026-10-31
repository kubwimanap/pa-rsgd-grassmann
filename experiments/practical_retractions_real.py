"""Practical inexact retractions on real data (digits and Hopkins 155 subset).

Part A: N+S (round-to-nearest t=8, minibatch b=8), constant step 0.02, K=4000:
        stationary error, drift and divergences for exact polar retraction,
        no re-orthonormalization, periodic re-orthonormalization (T=10), one
        Newton-Schulz step, landing (lambda=1) and PA-RSGD.
Part B: exact gradient and arithmetic: drift-stability edges of no
        re-orthonormalization and landing, bracketed on a step grid, against
        the predictions 1/mu_max and 1/(mu_max+lambda) of Corollary 1
        (mu_max = lambda_1 = 4 after scaling).  Newton-Schulz and periodic
        re-orthonormalization are bracketed on the same grid.

Outputs
  results/tables/practical_retractions_real.csv
  results/tables/practical_edges_real.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, digits, hopkins_problems, pa_rsgd, random_start, rsgd
from parsgd.util import parse, pmap, write_csv

DIG = digits()
HOP = hopkins_problems()
SCHEMES = {
    "exact polar retraction": Retraction("surrogate", 0.0),
    "no re-orthonormalization": Retraction("none"),
    "re-orthonormalize every 10 steps": Retraction("periodic", T=10),
    "one Newton-Schulz step": Retraction("ns"),
    "landing, lambda=1": Retraction("landing", lam=1.0),
    "PA-RSGD (no re-orthonormalization + drift monitor)": None,
}
GRID = np.geomspace(0.1, 0.6, 24)
EDGE_SCHEMES = {"no re-orthonormalization": (Retraction("none"), 0.25),
                "landing, lambda=1": (Retraction("landing", lam=1.0), 0.2),
                "re-orthonormalize every 10 steps": (Retraction("periodic", T=10), None),
                "one Newton-Schulz step": (Retraction("ns"), None)}


def problem(c):
    return DIG if c["data"] == "digits" else HOP[c["idx"]]


def job_a(c):
    P = problem(c)
    rng = np.random.default_rng(7100 + 37 * c["seed"] + c.get("idx", 0))
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(8, "RN", rng)
    t0 = time.perf_counter()
    if SCHEMES[c["scheme"]] is None:
        tr = pa_rsgd(P, X0, c["K"], 8, Retraction("none"), ar, rng, rec=13)
    else:
        tr = rsgd(P, X0, c["K"], 0.02, 8, SCHEMES[c["scheme"]], ar, rng, rec=13)
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"), div=bool(tr.diverged),
                sec=time.perf_counter() - t0)


def job_b(c):
    P = problem(c)
    rng = np.random.default_rng(9100 + 37 * c["seed"] + c.get("idx", 0))
    X0 = random_start(P.n, P.p, rng)
    tr = rsgd(P, X0, c["K"], c["eta"], None, EDGE_SCHEMES[c["scheme"]][0], Arith(), rng, rec=50)
    ok = (not tr.diverged) and tr.stationary("err") < 0.9 and tr.stationary("drift") < 1e-3
    return dict(c, stable=bool(ok))


def bracket(res):
    by = {}
    for r in res:
        by.setdefault(r["eta"], []).append(r["stable"])
    last = None
    for e in sorted(by):
        if all(by[e]):
            last = e
        else:
            return last, e
    return last, np.inf


def main():
    a = parse(__doc__)
    K = 1500 if a.quick else 4000
    sd = 3 if a.quick else 10
    sh = 1 if a.quick else 3
    hop = range(len(HOP)) if not a.quick else range(min(4, len(HOP)))
    cfgs = [dict(data="digits", scheme=k, seed=s, K=K) for k in SCHEMES for s in range(sd)]
    cfgs += [dict(data="hopkins", idx=i, scheme=k, seed=s, K=K) for k in SCHEMES for i in hop for s in range(sh)]
    res = pmap(job_a, cfgs, a.workers)
    rows = []
    for data in ("digits", "hopkins"):
        for k in SCHEMES:
            rr = [r for r in res if r["data"] == data and r["scheme"] == k]
            fin = [r for r in rr if not r["div"]]
            e = np.array([r["err"] for r in fin])
            rows.append(dict(data=data, scheme=k, runs=len(rr), diverged=sum(r["div"] for r in rr),
                             error_mean=e.mean() if len(e) else np.nan,
                             error_sd=e.std(ddof=1) if len(e) > 1 else np.nan,
                             drift_mean=np.mean([r["drift"] for r in fin]) if fin else np.nan,
                             drift_max=np.max([r["drift"] for r in fin]) if fin else np.nan,
                             ms_per_iter=1e3 * np.mean([r["sec"] for r in rr]) / K))
    write_csv("practical_retractions_real.csv", rows)

    Kb = 600 if a.quick else 1500
    grid = GRID[::3] if a.quick else GRID
    cfgs = [dict(data="digits", scheme=k, eta=e, seed=s, K=Kb) for k in EDGE_SCHEMES for e in grid for s in range(2)]
    hop_b = range(0, len(HOP), 3) if not a.quick else range(min(2, len(HOP)))
    cfgs += [dict(data="hopkins", idx=i, scheme=k, eta=e, seed=0, K=Kb) for k in EDGE_SCHEMES for e in grid
             for i in hop_b]
    res = pmap(job_b, cfgs, a.workers)
    er = []
    for data in ("digits", "hopkins"):
        for k, (_, pred) in EDGE_SCHEMES.items():
            lo, hi = bracket([r for r in res if r["data"] == data and r["scheme"] == k])
            er.append(dict(data=data, scheme=k, predicted_edge=pred if pred else "NA",
                           last_stable=lo, first_unstable=hi,
                           inside=(bool(lo is not None and lo <= pred <= hi) if pred else "NA")))
    write_csv("practical_edges_real.csv", er)


if __name__ == "__main__":
    main()
