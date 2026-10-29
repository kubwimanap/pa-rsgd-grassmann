"""Figure 2: observed stationary ||grad f||^2 against the a-priori floors of
Theorem 3 ((18) for exact arithmetic and SR, (19) for RN).

Stochastic configurations: 3 gaps x eta in {0.005,0.01,0.02,0.05} x
b in {8,32,128} x gamma in {0,0.9} x 5 arithmetic settings (exact; SR t=10,8;
RN t=10,8) = 360.  Full-gradient configurations: 3 gaps x 4 steps x 2 gammas
x 4 arithmetic settings = 96.  All runs start near the solution.

Outputs
  results/tables/fig2_floors.csv
  results/tables/fig2_summary.csv
  results/figures/fig2_floors.png/.pdf
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, near_solution, rsgd, synthetic
from parsgd.bounds import constants_at_solution, floor_bound
from parsgd.util import parse, pmap, savefig, write_csv

TAILS = (1.0, 2.0, 2.3)
PROBS = {tl: synthetic(0, tail=(0.2, tl)) for tl in TAILS}
ETAS = (0.005, 0.01, 0.02, 0.05)
ARITH = {"exact": (None, "RN"), "SR t=10": (10, "SR"), "SR t=8": (8, "SR"),
         "RN t=10": (10, "RN"), "RN t=8": (8, "RN")}


def job(c):
    P = PROBS[c["tail"]]
    rng = np.random.default_rng(11000 + c["seed"])
    X0 = near_solution(P, rng)
    t, mode = ARITH[c["arith"]]
    tr = rsgd(P, X0, c["K"], c["eta"], c["b"], Retraction("surrogate", c["gamma"]), Arith(t, mode, rng), rng,
              rec=25, want_g2=True)
    return dict(c, g2=tr.stationary("g2"), drift=tr.stationary("drift"))


def main():
    a = parse(__doc__)
    S = 1 if a.quick else 5
    K = 1500 if a.quick else 6000
    cfgs = [dict(tail=tl, eta=e, b=b, gamma=g, arith=ar, seed=s, K=K)
            for tl, e, b, g, ar in itertools.product(TAILS, ETAS, (8, 32, 128), (0.0, 0.9), ARITH) for s in range(S)]
    cfgs += [dict(tail=tl, eta=e, b=None, gamma=g, arith=ar, seed=s, K=K)
             for tl, e, g, ar in itertools.product(TAILS, ETAS, (0.0, 0.9), [k for k in ARITH if k != "exact"])
             for s in range(S)]
    res = pmap(job, cfgs, a.workers)

    groups = {}
    for r in res:
        groups.setdefault((r["tail"], r["eta"], r["b"], r["gamma"], r["arith"]), []).append(r)
    consts = {}
    rows = []
    for (tl, e, b, g, ar), rr in groups.items():
        P = PROBS[tl]
        if (tl, e, b) not in consts:
            consts[tl, e, b] = constants_at_solution(P, e, b)
        t, mode = ARITH[ar]
        Phi, D, vac = floor_bound(consts[tl, e, b], e, g, t, mode, P.p)
        obs = float(np.mean([r["g2"] for r in rr]))
        rows.append(dict(gap=round(P.gap, 3), eta=e, b="full" if b is None else b, gamma=g, arith=ar,
                         observed_g2=obs, bound=Phi, ratio=obs / Phi if np.isfinite(Phi) and Phi > 0 else np.nan,
                         predicted_drift=D, observed_drift=float(np.mean([r["drift"] for r in rr])),
                         vacuous=bool(vac), exceeded=bool(np.isfinite(Phi) and obs > Phi)))
    write_csv("fig2_floors.csv", rows)

    summ = []
    for name, sel in (("exact, stochastic", lambda r: r["arith"] == "exact"),
                      ("SR, stochastic", lambda r: r["arith"].startswith("SR") and r["b"] != "full"),
                      ("RN, stochastic", lambda r: r["arith"].startswith("RN") and r["b"] != "full"),
                      ("RN, full gradient", lambda r: r["arith"].startswith("RN") and r["b"] == "full"),
                      ("SR, full gradient", lambda r: r["arith"].startswith("SR") and r["b"] == "full")):
        rr = [r for r in rows if sel(r) and not r["vacuous"]]
        q = np.array([r["ratio"] for r in rr])
        q = q[np.isfinite(q)]
        summ.append(dict(group=name, configs=len(rr), ratio_min=q.min() if len(q) else np.nan,
                         ratio_median=np.median(q) if len(q) else np.nan, ratio_max=q.max() if len(q) else np.nan))
    summ.append(dict(group="all", configs=len(rows), vacuous=sum(r["vacuous"] for r in rows),
                     exceeded=sum(r["exceeded"] for r in rows)))
    write_csv("fig2_summary.csv", summ, ["group", "configs", "ratio_min", "ratio_median", "ratio_max",
                                          "vacuous", "exceeded"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = {"exact": "k", "SR": "C0", "RN": "C3"}
    ok = [r for r in rows if not r["vacuous"] and np.isfinite(r["bound"])]
    for key, col in cols.items():
        rr = [r for r in ok if r["arith"].startswith(key)]
        ax[0].loglog([r["bound"] for r in rr], [r["observed_g2"] for r in rr], "o", ms=3, color=col, label=key, alpha=0.7)
    lim = [min(r["bound"] for r in ok) / 3, max(r["bound"] for r in ok) * 3]
    ax[0].loglog(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="predicted floor (18)/(19)", ylabel="observed stationary ||grad f||^2", title="(a)")
    ax[0].legend(fontsize=8)
    data = [[r["ratio"] for r in ok if sel(r)] for _, sel in
            (("", lambda r: r["arith"] == "exact"), ("", lambda r: r["arith"].startswith("SR") and r["b"] != "full"),
             ("", lambda r: r["arith"].startswith("RN") and r["b"] != "full"),
             ("", lambda r: r["arith"].startswith("SR") and r["b"] == "full"),
             ("", lambda r: r["arith"].startswith("RN") and r["b"] == "full"))]
    ax[1].boxplot([d if d else [np.nan] for d in data])
    ax[1].set_xticks(range(1, 6))
    ax[1].set_xticklabels(["exact", "SR", "RN", "SR full", "RN full"])
    ax[1].set(yscale="log", ylabel="observed / predicted", title="(b)")
    f.tight_layout()
    savefig(f, "fig2_floors")


if __name__ == "__main__":
    main()
