"""Table 3: practical inexact retractions -- stationary drift, ratio to the
a-priori drift prediction, and drift-stability edges (from scheme_edges.csv).

Outputs
  results/tables/table3_practical_retractions.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, random_start, rsgd, synthetic
from parsgd.bounds import constants_at_solution, drift_prediction_scheme
from parsgd.util import parse, pmap, read_csv, write_csv

P = synthetic(0)
ETA = 0.02
SCHEMES = {
    "surrogate (3), gamma=0.9": Retraction("surrogate", 0.9),
    "no re-orthonormalization": Retraction("none"),
    "re-orthonormalize every 10 steps": Retraction("periodic", T=10),
    "re-orthonormalize every 50 steps": Retraction("periodic", T=50),
    "one Newton-Schulz step": Retraction("ns"),
    "landing, lambda=0.5": Retraction("landing", lam=0.5),
    "landing, lambda=1.0": Retraction("landing", lam=1.0),
    "landing, lambda=2.0": Retraction("landing", lam=2.0),
}


def job(c):
    rng = np.random.default_rng(3000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    tr = rsgd(P, X0, c["K"], ETA, c["b"], SCHEMES[c["scheme"]], Arith(c["t"], "RN", rng), rng, rec=13)
    return dict(c, drift=tr.stationary("drift"))


def main():
    a = parse(__doc__)
    S = 3 if a.quick else 10
    K = 1500 if a.quick else 4000
    cfgs = [dict(scheme=k, b=b, t=t, seed=s, K=K) for k in SCHEMES for b in (8, 32) for t in (None, 10)
            for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    consts = {b: constants_at_solution(P, ETA, b) for b in (8, 32)}
    edges = {r["scheme"]: r for r in (read_csv("scheme_edges.csv") or [])}
    rows = []
    for k, R in SCHEMES.items():
        row = dict(scheme=k)
        ratios = []
        for b in (8, 32):
            for t in (None, 10):
                d = np.mean([r["drift"] for r in res if r["scheme"] == k and r["b"] == b and r["t"] == t])
                u = 0.0 if t is None else 2.0 ** -t
                pred = drift_prediction_scheme(consts[b], ETA, R, u, P.p)
                ratios.append(d / pred)
                if t is None:
                    row[f"drift_b{b}"] = d
        row["ratio_min"], row["ratio_max"] = min(ratios), max(ratios)
        e = edges.get(k)
        if e:
            row["predicted_edge"] = e["predicted_edge"]
            row["empirical_edge"] = f"[{float(e['last_stable']):.3f}, {float(e['first_unstable']):.3f})"
        rows.append(row)
    write_csv("table3_practical_retractions.csv", rows,
              ["scheme", "drift_b8", "drift_b32", "ratio_min", "ratio_max", "predicted_edge", "empirical_edge"])


if __name__ == "__main__":
    main()
