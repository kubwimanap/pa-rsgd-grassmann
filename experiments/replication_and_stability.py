"""Table S4 and the Section 3.2 stability checks.

* Table S4: all eight perturbation cells for the robust projection energy
  (N: t=7, S: b=32, G: gamma=0.95; 4 seeds) and the Hopkins 155 subset
  (N: t=7, S: b=8, G: gamma=0.9; 3 seeds; only if the data are present).
* Divergence through drift interaction at eta = 0.3, b = 16.
* Reconstruction error with X X^T instead of the orthogonal projector.
* Damping assumption (Proposition 1) with the non-damped extension F2.

Outputs
  results/tables/tableS4_replication.csv
  results/tables/divergence_interaction.csv
  results/tables/reconstruction_drift.csv
  results/tables/damping_F2.csv
  results/tables/hopkins_methods.csv   (only if the Hopkins data are present)
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import (Arith, F2Extension, PAConfig, Retraction, Robust, hopkins_problems, orth,
                    pa_rsgd, random_start, rsgd, synthetic)
from parsgd.algorithms import Trace, apply_retraction, direction, drift
from parsgd.util import parse, pmap, write_csv

P = synthetic(0)
ROBUST = Robust(0)
HOPKINS = hopkins_problems()
CELLS = {"clean": (0, 0, 0), "N": (1, 0, 0), "S": (0, 1, 0), "G": (0, 0, 1), "N+S": (1, 1, 0),
         "N+G": (1, 0, 1), "S+G": (0, 1, 1), "N+S+G": (1, 1, 1)}


def cell_job(c):
    prob = ROBUST if c["prob"] == "robust" else HOPKINS[c["idx"]]
    i, j, l = CELLS[c["cell"]]
    rng = np.random.default_rng(4000 + c["seed"])
    X0 = random_start(prob.n, prob.p, rng)
    tr = rsgd(prob, X0, c["K"], 0.02, c["b"] if j else None,
              Retraction("surrogate", c["gamma"] if l else 0.0), Arith(c["t"] if i else None, "RN", rng), rng,
              rec=13)
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"))


def div_job(c):
    rng = np.random.default_rng(6000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    R = Retraction("surrogate", c["gamma"])
    if c["method"] == "pa":
        tr = pa_rsgd(P, X0, c["K"], c["b"], R, Arith(), rng, eta_init=0.3)
    else:
        tr = rsgd(P, X0, c["K"], 0.3, c["b"], R, Arith(), rng)
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"), div=tr.diverged)


def recon_job(c):
    rng = np.random.default_rng(6500 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    X, ar = X0.copy(), Arith()
    for k in range(c["K"]):
        d = direction(P, X, None, ar)
        X = apply_retraction(Retraction("surrogate", 0.99), X - c["eta"] * d, k, ar)
    test = P.sample(np.random.default_rng(1), 2000)
    Q = orth(X)
    e_proj = np.linalg.norm(test - Q @ (Q.T @ test)) ** 2
    e_xx = np.linalg.norm(test - X @ (X.T @ test)) ** 2
    return dict(c, drift=drift(X), increase_pct=100 * (e_xx / e_proj - 1))


def hop_job(c):
    prob = HOPKINS[c["idx"]]
    rng = np.random.default_rng(4500 + c["seed"])
    X0 = random_start(prob.n, prob.p, rng)
    ar, R = Arith(8, "RN", rng), Retraction("surrogate", 0.9)
    if c["method"] == "PA-RSGD":
        tr = pa_rsgd(prob, X0, c["K"], 8, R, ar, rng)
    else:
        tr = rsgd(prob, X0, c["K"], 0.02, 8, R, ar, rng, decay=200 if "decaying" in c["method"] else None)
    return dict(c, err=tr.stationary("err"))


def f2_job(c):
    prob = F2Extension(P, c["s"])
    rng = np.random.default_rng(6800 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    tr = rsgd(prob, X0, c["K"], 0.02, None, Retraction("surrogate", c["gamma"]), Arith(), rng)
    return dict(c, div=tr.diverged or tr.stationary("drift") > 1.0, drift=tr.stationary("drift"))


def main():
    a = parse(__doc__)
    K = 1500 if a.quick else 4000

    # ---- Table S4 ---------------------------------------------------------
    cfgs = [dict(prob="robust", cell=c, t=7, b=32, gamma=0.95, seed=s, K=K)
            for c in CELLS for s in range(2 if a.quick else 4)]
    for idx in range(len(HOPKINS)):
        cfgs += [dict(prob="hopkins", idx=idx, cell=c, t=7, b=8, gamma=0.9, seed=s, K=K)
                 for c in CELLS for s in range(2 if a.quick else 3)]
    res = pmap(cell_job, cfgs, a.workers)
    rows = []
    for c in CELLS:
        row = dict(case=c)
        rr = [r for r in res if r["prob"] == "robust" and r["cell"] == c]
        row["robust_error"], row["robust_drift"] = np.mean([r["err"] for r in rr]), np.mean([r["drift"] for r in rr])
        hh = [r for r in res if r["prob"] == "hopkins" and r["cell"] == c]
        row["hopkins_error"] = np.mean([r["err"] for r in hh]) if hh else "not run (data missing)"
        row["hopkins_drift"] = np.mean([r["drift"] for r in hh]) if hh else ""
        rows.append(row)
    write_csv("tableS4_replication.csv", rows)

    # ---- Hopkins subset under N+S+G (t=8, b=8, gamma=0.9): Section 3.3 ----------
    if HOPKINS:
        meths = ("RSGD, fixed step", "RSGD, decaying step", "PA-RSGD")
        res = pmap(hop_job, [dict(idx=i, method=m, seed=s, K=K) for i in range(len(HOPKINS)) for m in meths
                             for s in range(2 if a.quick else 3)], a.workers)
        from parsgd.util import mean_ci
        write_csv("hopkins_methods.csv",
                  [dict(method=m, problems=len(HOPKINS),
                        error_mean=float(np.mean([r["err"] for r in res if r["method"] == m])),
                        error_sd=float(np.std([r["err"] for r in res if r["method"] == m], ddof=1)))
                   for m in meths])

    # ---- divergence through interaction -------------------------------------
    S = 3 if a.quick else 10
    cfgs = [dict(case="noise only (gamma=0, b=16)", method="rsgd", gamma=0.0, b=16, seed=s, K=1500) for s in range(S)]
    cfgs += [dict(case="defect only (gamma=0.99, exact gradient)", method="rsgd", gamma=0.99, b=None, seed=s, K=1500)
             for s in range(S)]
    cfgs += [dict(case="both (gamma=0.99, b=16)", method="rsgd", gamma=0.99, b=16, seed=s, K=1500) for s in range(S)]
    cfgs += [dict(case="both, PA-RSGD", method="pa", gamma=0.99, b=16, seed=s, K=1500) for s in range(S)]
    res = pmap(div_job, cfgs, a.workers)
    out = []
    for case in dict.fromkeys(r["case"] for r in res):
        rr = [r for r in res if r["case"] == case]
        fin = [r for r in rr if not r["div"]]
        out.append(dict(case=case, diverged=f"{sum(r['div'] for r in rr)}/{len(rr)}",
                        error_finite_runs=np.mean([r["err"] for r in fin]) if fin else np.nan,
                        drift_finite_runs=np.mean([r["drift"] for r in fin]) if fin else np.nan))
    write_csv("divergence_interaction.csv", out)

    # ---- reconstruction with X X^T ---------------------------------------
    res = pmap(recon_job, [dict(eta=e, seed=s, K=1500) for e in (0.02, 0.3, 0.4) for s in range(2 if a.quick else 3)],
               a.workers)
    write_csv("reconstruction_drift.csv",
              [dict(eta=e, drift=np.mean([r["drift"] for r in res if r["eta"] == e]),
                    reconstruction_increase_pct=np.mean([r["increase_pct"] for r in res if r["eta"] == e]))
               for e in (0.02, 0.3, 0.4)])

    # ---- damping assumption with F2 -----------------------------------------
    res = pmap(f2_job, [dict(s=s_, gamma=g, seed=i, K=K) for s_ in (1.0, 0.2) for g in (0.9, 0.99) for i in range(6)],
               a.workers)
    write_csv("damping_F2.csv",
              [dict(C=f"{s_}A", gamma=g, diverged=f"{sum(r['div'] for r in res if r['s'] == s_ and r['gamma'] == g)}/6")
               for s_ in (1.0, 0.2) for g in (0.9, 0.99)])


if __name__ == "__main__":
    main()
