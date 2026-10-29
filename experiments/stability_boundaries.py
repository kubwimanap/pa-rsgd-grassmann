"""Table S1 and Figure 3: largest stable constant step vs the retraction defect,
plus the drift-stability edges of the practical schemes used in Table 3.

A run is stable if it does not diverge, its error stays below 0.9 and its
drift below 1e-3 (exact gradient) or 0.3 (b = 32).

Outputs
  results/tables/tableS1_stability.csv
  results/tables/scheme_edges.csv
  results/figures/fig3_stability.png/.pdf
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, random_start, rsgd, synthetic
from parsgd.bounds import surrogate_edge
from parsgd.util import parse, pmap, savefig, write_csv

P = synthetic(0)
GRID = np.geomspace(0.1, 1.2, 40)


def job(c):
    rng = np.random.default_rng(9000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    R = c["R"]
    tr = rsgd(P, X0, c["K"], c["eta"], c["b"], R, Arith(), rng, rec=50)
    dmax = 1e-3 if c["b"] is None else 0.3
    ok = (not tr.diverged) and tr.stationary("err") < 0.9 and tr.stationary("drift") < dmax
    return dict(c, stable=bool(ok))


def bracket(res):
    """Last grid value stable for every seed before the first unstable one."""
    by = {}
    for r in res:
        by.setdefault(r["eta"], []).append(r["stable"])
    etas = sorted(by)
    last = None
    for e in etas:
        if all(by[e]):
            last = e
        else:
            return last, e
    return last, np.inf


def mu_max_minibatch(b, n_mc=2000, seed=7):
    rng = np.random.default_rng(seed)
    U = P.Up
    vals = []
    for _ in range(n_mc):
        S = P.sample(rng, b)
        H = U.T @ S @ (S.T @ U) / b
        vals.append(np.linalg.eigvalsh(H).max())
    return float(np.mean(vals))


def main():
    a = parse(__doc__)
    S = 2 if a.quick else 3
    K = 600 if a.quick else 1500
    grid = GRID[::2] if a.quick else GRID
    gammas = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    cfgs = [dict(gamma=g, b=b, R=Retraction("surrogate", g), eta=e, K=K, seed=s)
            for g in gammas for b in (None, 32) for e in grid for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    mu_exact = float(P.lam[0])
    mu_32 = mu_max_minibatch(32)
    rows = []
    for g in gammas:
        row = dict(gamma=g)
        for b, mu, tag in ((None, mu_exact, "exact"), (32, mu_32, "b32")):
            lo, hi = bracket([r for r in res if r["gamma"] == g and r["b"] == b])
            row[f"{tag}_last_stable"], row[f"{tag}_first_unstable"] = lo, hi
            row[f"{tag}_theorem1_edge"] = surrogate_edge(g, mu)
        rows.append(row)
    write_csv("tableS1_stability.csv", rows)

    # practical schemes (exact gradient): no re-orthonormalisation and landing
    schemes = {"no re-orthonormalization": (Retraction("none"), 1 / mu_exact)}
    for lam in (0.5, 1.0, 2.0):
        schemes[f"landing, lambda={lam}"] = (Retraction("landing", lam=lam), 1 / (mu_exact + lam))
    cfgs = [dict(scheme=k, R=v[0], b=None, eta=e, K=K, seed=s) for k, v in schemes.items()
            for e in grid for s in range(S)]
    res2 = pmap(job, cfgs, a.workers)
    er = []
    for k, (_, pred) in schemes.items():
        lo, hi = bracket([r for r in res2 if r["scheme"] == k])
        er.append(dict(scheme=k, predicted_edge=pred, last_stable=lo, first_unstable=hi,
                       inside=bool(lo is not None and lo <= pred <= hi)))
    write_csv("scheme_edges.csv", er)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    gg = np.linspace(0.25, 0.995, 200)
    for tag, mu, col, lab in (("exact", mu_exact, "C0", "exact gradient"), ("b32", mu_32, "C1", "b = 32")):
        for r in rows:
            lo, hi = r[f"{tag}_last_stable"], r[f"{tag}_first_unstable"]
            if lo is None or not np.isfinite(hi):
                continue
            ax.errorbar(r["gamma"], 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt="o", color=col, ms=4)
        ax.plot(gg, [surrogate_edge(g, mu) for g in gg], "-", color=col, label=f"eta_G(gamma), {lab}")
    ax.set(xlabel="defect gamma", ylabel="largest stable step", ylim=(0, 1.3))
    ax.legend(fontsize=8)
    savefig(fig, "fig3_stability")


if __name__ == "__main__":
    main()
