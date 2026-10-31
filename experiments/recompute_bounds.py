"""Recompute the a-priori bounds of Figure 2 and Figure 4(c) from the stored
observations (results/tables/fig2_floors.csv, fig4c_drift_vs_bound.csv) after a
change of the bound formulas in parsgd/bounds.py, without re-running the
simulations.  Rewrites fig2_floors.csv, fig2_summary.csv and
fig4c_drift_vs_bound.csv and redraws Figures 2 and 4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import convergence_floors as CF
from parsgd import synthetic
from parsgd.bounds import constants_at_solution, drift_bounds, floor_bound
from parsgd.util import read_csv, write_csv


def main():
    rows = read_csv("fig2_floors.csv")
    tail_of = {round(P.gap, 3): tl for tl, P in CF.PROBS.items()}
    consts, out = {}, []
    for r in rows:
        tl = tail_of[float(r["gap"])]
        P = CF.PROBS[tl]
        e = float(r["eta"])
        b = None if r["b"] == "full" else int(r["b"])
        g = float(r["gamma"])
        if (tl, e, b) not in consts:
            consts[tl, e, b] = constants_at_solution(P, e, b)
        t, mode = CF.ARITH[r["arith"]]
        Phi, D, vac = floor_bound(consts[tl, e, b], e, g, t, mode, P.p)
        obs = float(r["observed_g2"])
        out.append(dict(gap=r["gap"], eta=e, b=r["b"], gamma=g, arith=r["arith"], observed_g2=obs, bound=Phi,
                        ratio=obs / Phi if np.isfinite(Phi) and Phi > 0 else np.nan, predicted_drift=D,
                        observed_drift=float(r["observed_drift"]), vacuous=bool(vac),
                        exceeded=bool(np.isfinite(Phi) and obs > Phi)))
    write_csv("fig2_floors.csv", out)
    summ = []
    for name, sel in (("exact, stochastic", lambda r: r["arith"] == "exact"),
                      ("SR, stochastic", lambda r: r["arith"].startswith("SR") and r["b"] != "full"),
                      ("RN, stochastic", lambda r: r["arith"].startswith("RN") and r["b"] != "full"),
                      ("RN, full gradient", lambda r: r["arith"].startswith("RN") and r["b"] == "full"),
                      ("SR, full gradient", lambda r: r["arith"].startswith("SR") and r["b"] == "full")):
        rr = [r for r in out if sel(r) and not r["vacuous"]]
        q = np.array([r["ratio"] for r in rr])
        q = q[np.isfinite(q)]
        summ.append(dict(group=name, configs=len(rr), ratio_min=q.min() if len(q) else np.nan,
                         ratio_median=np.median(q) if len(q) else np.nan, ratio_max=q.max() if len(q) else np.nan))
    summ.append(dict(group="all", configs=len(out), vacuous=sum(r["vacuous"] for r in out),
                     exceeded=sum(r["exceeded"] for r in out)))
    write_csv("fig2_summary.csv", summ, ["group", "configs", "ratio_min", "ratio_median", "ratio_max",
                                          "vacuous", "exceeded"])

    P = synthetic(0)
    fc, cache = [], {}
    for r in read_csv("fig4c_drift_vs_bound.csv"):
        b, g = int(r["b"]), float(r["gamma"])
        t = None if r["t"] == "inf" else int(r["t"])
        if b not in cache:
            cache[b] = constants_at_solution(P, 0.02, b)
        u = 0.0 if t is None else 2.0 ** -t
        D2 = drift_bounds(cache[b], 0.02, g, u, P.p)[1]
        obs = float(r["drift_obs"])
        fc.append(dict(gamma=g, b=b, t=r["t"], drift_obs=obs, bound_D2=D2, ratio=obs / D2))
    write_csv("fig4c_drift_vs_bound.csv", fc)

    from make_figures import fig2, fig4
    fig2()
    fig4()


if __name__ == "__main__":
    main()
