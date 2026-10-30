"""Figure S1: optical digits (p = 5), error versus iteration, median and
interquartile range over 5 seeds.  (a) stochastic noise only, (b) N+S+G.
The tuned constant step is read from table6_tuned_steps.csv (run baselines.py first).

Outputs
  results/tables/figS1_digits_traces.csv
  results/figures/figS1_digits.png/.pdf
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, PAConfig, Retraction, digits, pa_rsgd, random_start, rsgd
from parsgd.util import parse, pmap, read_csv, write_csv

P = digits(5)
PANELS = {"(a) Digits, S only": dict(t=None, b=8, gamma=0.0), "(b) Digits, N+S+G": dict(t=8, b=8, gamma=0.9)}
METHODS = ("RSGD, decaying", "RSGD, oracle-tuned eta", "PA-RSGD", "PA-RSGD, anytime horizon")


def job(c):
    sc = PANELS[c["panel"]]
    rng = np.random.default_rng(200 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    ar, R = Arith(sc["t"], "RN", rng), Retraction("surrogate", sc["gamma"])
    m = c["method"]
    if m == "RSGD, decaying":
        tr = rsgd(P, X0, c["K"], 0.02, sc["b"], R, ar, rng, decay=200)
    elif m == "RSGD, oracle-tuned eta":
        tr = rsgd(P, X0, c["K"], c["eta"], sc["b"], R, ar, rng)
    else:
        tr = pa_rsgd(P, X0, c["K"], sc["b"], R, ar, rng, cfg=PAConfig(anytime="anytime" in m))
    return dict(c, it=tr.it, err=tr.err)


def main():
    a = parse(__doc__)
    K = 1500 if a.quick else 4000
    tuned = [r for r in (read_csv("table6_tuned_steps.csv") or [])
             if r["scenario"] == "Digits N+S+G" and r["method"] == "rsgd"]
    eta = float(tuned[0]["tuned_eta"]) if tuned else 0.05
    res = pmap(job, [dict(panel=p, method=m, eta=eta, seed=s, K=K) for p in PANELS for m in METHODS
                     for s in range(5)], a.workers)

    rows = []
    for p in PANELS:
        for m in METHODS:
            rr = [r for r in res if r["panel"] == p and r["method"] == m]
            n = min(len(r["err"]) for r in rr)
            it = np.asarray(rr[0]["it"][:n])
            E = np.array([r["err"][:n] for r in rr])
            q1, med, q3 = np.percentile(E, [25, 50, 75], axis=0)
            rows += [dict(panel=p, method=m, iteration=int(it[j]), median=med[j], q25=q1[j], q75=q3[j])
                     for j in range(n)]
    write_csv("figS1_digits_traces.csv", rows)
    from make_figures import figS1

    figS1()

if __name__ == "__main__":
    main()
