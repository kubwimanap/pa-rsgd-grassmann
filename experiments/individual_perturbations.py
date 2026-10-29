"""Table S3 and Figure 4: individual perturbations, scaling laws and drift vs bound.

Outputs
  results/tables/tableS3_individual.csv
  results/tables/fig4a_precision_floor.csv, fig4b_stochastic_floor.csv, fig4c_drift_vs_bound.csv
  results/tables/sr_vs_rn_t7.csv
  results/figures/fig4_scaling.png/.pdf
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, random_start, rsgd, synthetic
from parsgd.bounds import constants_at_solution, drift_bounds
from parsgd.util import mean_ci, parse, pmap, savefig, write_csv

P = synthetic(0)


def job(c):
    rng = np.random.default_rng(1000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(c.get("t"), c.get("mode", "RN"), rng)
    tr = rsgd(P, X0, c["K"], c["eta"], c.get("b"), Retraction("surrogate", c.get("gamma", 0.0)), ar, rng)
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"), div=tr.diverged)


def agg(res, keys):
    out = {}
    for r in res:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def main():
    a = parse(__doc__)
    S = 2 if a.quick else 5
    K = 1500 if a.quick else 4000
    eta = 0.02

    # ---------------- Table S3 -----------------------------------------
    cfgs = []
    for t in range(12, 5, -1):
        cfgs += [dict(pert="N", level=f"t={t}", t=t, eta=eta, K=K, seed=s) for s in range(S)]
    for b in (128, 64, 32, 16, 8, 4):
        cfgs += [dict(pert="S", level=f"b={b}", b=b, eta=eta, K=K, seed=s) for s in range(S)]
    for g in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
        cfgs += [dict(pert="G", level=f"gamma={g}", gamma=g, eta=eta, K=K, seed=s) for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    rows = []
    for (pert, level), rs in agg(res, ("pert", "level")).items():
        e = np.array([r["err"] for r in rs]); d = np.array([r["drift"] for r in rs])
        rows.append(dict(perturbation=pert, level=level, err_mean=e.mean(), err_sd=e.std(ddof=1),
                         drift_mean=d.mean(), seeds=len(rs)))
    write_csv("tableS3_individual.csv", rows)

    # ---------------- SR vs RN at t = 7, no noise ------------------------
    res = pmap(job, [dict(t=7, mode=m, eta=eta, K=K, seed=s) for m in ("RN", "SR") for s in range(S)], a.workers)
    write_csv("sr_vs_rn_t7.csv", [dict(mode=m, err_mean=np.mean([r["err"] for r in res if r["mode"] == m]),
                                       seeds=S) for m in ("RN", "SR")])

    # ---------------- Figure 4a: precision floor vs u/eta ------------------
    cfgs = [dict(t=t, eta=e, K=K, seed=s) for t in range(7, 13) for e in (0.01, 0.02, 0.05) for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    fa = []
    for (t, e), rs in agg(res, ("t", "eta")).items():
        fa.append(dict(t=t, eta=e, u_over_eta=2.0 ** -t / e, err=np.mean([r["err"] for r in rs])))
    write_csv("fig4a_precision_floor.csv", fa)

    # ---------------- Figure 4b: squared stochastic floor vs eta/b --------
    cfgs = [dict(b=b, eta=e, K=K, seed=s) for b in (8, 16, 32, 64) for e in (0.005, 0.01, 0.02, 0.05) for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    fb = []
    for (b, e), rs in agg(res, ("b", "eta")).items():
        fb.append(dict(b=b, eta=e, eta_over_b=e / b, err2=np.mean([r["err"] ** 2 for r in rs])))
    write_csv("fig4b_stochastic_floor.csv", fb)

    # ---------------- Figure 4c: drift vs a-priori bound ------------------
    cfgs = [dict(gamma=g, b=b, t=t, eta=eta, K=K, seed=s)
            for g in (0.5, 0.8, 0.9, 0.95) for b in (8, 16, 32, 64) for t in (None, 10) for s in range(S)]
    res = pmap(job, cfgs, a.workers)
    fc, consts = [], {}
    for (g, b, t), rs in agg(res, ("gamma", "b", "t")).items():
        if b not in consts:
            consts[b] = constants_at_solution(P, eta, b)
        u = 0.0 if t is None else 2.0 ** -t
        D2 = drift_bounds(consts[b], eta, g, u, P.p)[1]
        obs = np.mean([r["drift"] for r in rs])
        fc.append(dict(gamma=g, b=b, t="inf" if t is None else t, drift_obs=obs, bound_D2=D2, ratio=obs / D2))
    write_csv("fig4c_drift_vs_bound.csv", fc)

    # ---------------- figure -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.array([r["u_over_eta"] for r in fa]); y = np.array([r["err"] for r in fa])
    sl = np.polyfit(np.log(x), np.log(y), 1)[0]
    ax[0].loglog(x, y, "o", ms=4)
    ax[0].set(xlabel="u / eta", ylabel="stationary ||sin Theta||_F", title=f"(a) precision floor (slope {sl:.2f})")
    x = np.array([r["eta_over_b"] for r in fb]); y = np.array([r["err2"] for r in fb])
    sl = np.polyfit(np.log(x), np.log(y), 1)[0]
    ax[1].loglog(x, y, "o", ms=4)
    ax[1].set(xlabel="eta / b", ylabel="squared stochastic floor", title=f"(b) stochastic floor (slope {sl:.2f})")
    x = np.array([r["bound_D2"] for r in fc]); y = np.array([r["drift_obs"] for r in fc])
    ax[2].loglog(x, y, "o", ms=4)
    lim = [min(x.min(), y.min()) / 2, max(x.max(), y.max()) * 2]
    ax[2].loglog(lim, lim, "k--", lw=1)
    ax[2].set(xlabel="a-priori bound D_inf^(2)", ylabel="observed drift", title="(c) drift vs Theorem 1(iii)")
    fig.tight_layout()
    savefig(fig, "fig4_scaling")


if __name__ == "__main__":
    main()
