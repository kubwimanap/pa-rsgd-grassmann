"""Table 4, Table S2 and Figure 5: 2^3 factorial interaction contrasts.

Each perturbation is off (0) or on (1):
  N: round-to-nearest with t bits      (off: exact arithmetic)
  S: minibatch of size b               (off: exact gradient)
  G: inexact retraction                (off: exact polar retraction)
Common random seeds are used across the eight cells.

Outputs
  results/tables/table4_contrasts.csv
  results/tables/tableS2_additivity.csv
  results/figures/fig5_additivity.png/.pdf
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, random_start, rsgd, synthetic
from parsgd.util import bootstrap_ci, parse, pmap, write_csv

P = synthetic(0)
LEVELS = {
    "A, surrogate": dict(t=7, b=32, G=Retraction("surrogate", 0.9)),
    "B, surrogate": dict(t=10, b=8, G=Retraction("surrogate", 0.98)),
    "A, periodic (10)": dict(t=7, b=32, G=Retraction("periodic", T=10)),
    "A, landing (0.5)": dict(t=7, b=32, G=Retraction("landing", lam=0.5)),
    "A, Newton-Schulz": dict(t=7, b=32, G=Retraction("ns")),
}


def job(c):
    lv = LEVELS[c["level"]]
    i, j, l = c["cell"]
    rng = np.random.default_rng(5000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(lv["t"] if i else None, "RN", rng)
    R = lv["G"] if l else Retraction("surrogate", 0.0)
    tr = rsgd(P, X0, c["K"], 0.02, lv["b"] if j else None, R, ar, rng, rec=13)
    return dict(c, err=tr.stationary("err"), drift=tr.stationary("drift"))


def contrasts(y):
    """y[(i,j,l)] -> arrays over replicates; returns the four contrasts of (9)."""
    C = {}
    C["NS"] = y[1, 1, 0] - y[1, 0, 0] - y[0, 1, 0] + y[0, 0, 0]
    C["NG"] = y[1, 0, 1] - y[1, 0, 0] - y[0, 0, 1] + y[0, 0, 0]
    C["SG"] = y[0, 1, 1] - y[0, 1, 0] - y[0, 0, 1] + y[0, 0, 0]
    C["NSG"] = sum((-1) ** (3 - i - j - l) * y[i, j, l] for i, j, l in itertools.product((0, 1), repeat=3))
    return C


def main():
    a = parse(__doc__)
    R = 3 if a.quick else 8
    K = 1500 if a.quick else 4000
    cells = list(itertools.product((0, 1), repeat=3))
    cfgs = [dict(level=lv, cell=c, seed=s, K=K) for lv in LEVELS for c in cells for s in range(R)]
    res = pmap(job, cfgs, a.workers)

    t4, s2, fig = [], [], {}
    for lv, spec in LEVELS.items():
        y = {q: {} for q in ("err", "drift")}
        for q in y:
            for c in cells:
                y[q][c] = np.array([r[q] for r in res if r["level"] == lv and r["cell"] == c])
        for q in ("err", "drift"):
            row = dict(level=lv, quantity="error" if q == "err" else "drift", replicates=R)
            for name, v in contrasts(y[q]).items():
                m, lo, hi = bootstrap_ci(v)
                row[f"C_{name}"], row[f"C_{name}_lo"], row[f"C_{name}_hi"] = m, lo, hi
            t4.append(row)
        if lv.endswith("surrogate"):
            e = {c: y["err"][c].mean() for c in cells}
            d = {c: y["drift"][c].mean() for c in cells}
            u, gap = 2.0 ** -spec["t"], P.gap
            dith = lambda: np.sqrt(e[0, 1, 0] ** 2 + u * u * P.p / (6 * 0.02 * gap))
            combos = {"N+S": (1, 1, 0), "N+G": (1, 0, 1), "S+G": (0, 1, 1), "N+S+G": (1, 1, 1)}
            single = {"N": (1, 0, 0), "S": (0, 1, 0), "G": (0, 0, 1)}
            for name, c in combos.items():
                parts = [single[k] for k in name.split("+")]
                add_e = sum(e[p] for p in parts)
                add_d = sum(d[p] for p in parts)
                if "N" in name and "S" in name:
                    dp = dith()
                elif name == "S+G":
                    dp = e[0, 1, 0]
                else:
                    dp = e[1, 0, 0]
                s2.append(dict(level=lv[0], combination=name, observed_error=e[c], additive_prediction=add_e,
                               dithering_prediction=dp, I_err=e[c] / add_e, observed_drift=d[c],
                               additive_drift=add_d, replicates=R))
            fig[lv[0]] = (e, d)
    write_csv("table4_contrasts.csv", t4)
    write_csv("tableS2_additivity.csv", s2)

    from make_figures import fig5

    fig5()

if __name__ == "__main__":
    main()
