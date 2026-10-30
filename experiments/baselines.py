"""Table 6: PA-RSGD against tuned baselines (Figure S1 is in figS1_digits.py).

Oracle-tuned steps are selected per scenario from a 10-point log grid in
[0.003, 0.3] on separate tuning seeds and evaluated on separate test seeds.

Outputs
  results/tables/table6_baselines.csv
  results/tables/table6_tuned_steps.csv
  results/tables/table6_transfer.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import (Arith, PAConfig, Retraction, digits, pa_rsgd, random_start, rsgd, rsvrg,
                    synthetic)
from parsgd.util import mean_ci, parse, pmap, write_csv

PROB = {"synthetic": synthetic(0), "digits": digits(5)}
SCEN = {
    "Synthetic N": dict(prob="synthetic", t=7, b=None, gamma=0.0),
    "Synthetic S": dict(prob="synthetic", t=None, b=16, gamma=0.0),
    "Synthetic N+S": dict(prob="synthetic", t=7, b=16, gamma=0.0),
    "Synthetic N+S+G": dict(prob="synthetic", t=7, b=16, gamma=0.95),
    "Digits N+S+G": dict(prob="digits", t=8, b=8, gamma=0.9),
}
GRID = np.geomspace(0.003, 0.3, 10)
TUNE_SEEDS = range(100, 105)


def run(method, sc, eta, seed, K, mode="RN"):
    P = PROB[sc["prob"]]
    rng = np.random.default_rng(seed)
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(sc["t"], mode, rng)
    R = Retraction("surrogate", sc["gamma"])
    if method == "rsgd":
        return rsgd(P, X0, K, eta, sc["b"], R, ar, rng)
    if method == "decay":
        return rsgd(P, X0, K, 0.02, sc["b"], R, ar, rng, decay=200)
    if method == "oja":
        return rsgd(P, X0, K, eta, sc["b"], Retraction("none"), ar, rng)
    if method == "landing":
        return rsgd(P, X0, K, eta, sc["b"], Retraction("landing", lam=1.0), ar, rng)
    if method == "rsvrg":
        return rsvrg(P, X0, K, eta, sc["b"], R, ar, rng)
    if method == "pa":
        return pa_rsgd(P, X0, K, sc["b"], R, ar, rng)
    if method == "pa_anytime":
        return pa_rsgd(P, X0, K, sc["b"], R, ar, rng, cfg=PAConfig(anytime=True))
    raise ValueError(method)


def job(c):
    tr = run(c["method"], SCEN[c["scenario"]], c.get("eta"), c["seed"], c["K"], c.get("mode", "RN"))
    out = dict(c, err=tr.stationary("err"), div=tr.diverged)
    if c.get("trace"):
        out["trace"] = (tr.it, tr.err)
    return out


TUNED = ("rsgd", "oja", "landing", "rsvrg")


def main():
    a = parse(__doc__)
    K = 1500 if a.quick else 4000
    test = range(200, 203) if a.quick else range(200, 210)
    tune = range(100, 102) if a.quick else TUNE_SEEDS

    # ---- tuning -------------------------------------------------------
    cfgs = []
    for s, sc in SCEN.items():
        for m in TUNED:
            if m == "rsvrg" and sc["prob"] != "digits":
                continue
            cfgs += [dict(scenario=s, method=m, eta=e, seed=i, K=K) for e in GRID for i in tune]
    res = pmap(job, cfgs, a.workers)
    best = {}
    for s in SCEN:
        for m in TUNED:
            errs = {e: np.mean([r["err"] for r in res if r["scenario"] == s and r["method"] == m and r["eta"] == e])
                    for e in GRID}
            errs = {e: v for e, v in errs.items() if np.isfinite(v)}
            if errs:
                best[s, m] = min(errs, key=errs.get)
    write_csv("table6_tuned_steps.csv", [dict(scenario=s, method=m, tuned_eta=e) for (s, m), e in best.items()])

    # ---- evaluation on test seeds ----------------------------------------
    methods = {
        "RSGD, eta=0.02": ("rsgd", lambda s: 0.02, "RN"),
        "RSGD, decaying": ("decay", lambda s: None, "RN"),
        "RSGD, oracle-tuned eta": ("rsgd", lambda s: best[s, "rsgd"], "RN"),
        "RSGD, tuned eta, SR": ("rsgd", lambda s: best[s, "rsgd"], "SR"),
        "Oja (tuned eta)": ("oja", lambda s: best[s, "oja"], "RN"),
        "Landing (tuned eta, lambda=1)": ("landing", lambda s: best[s, "landing"], "RN"),
        "PA-RSGD": ("pa", lambda s: None, "RN"),
        "PA-RSGD, anytime horizon": ("pa_anytime", lambda s: None, "RN"),
        "PA-RSGD, SR": ("pa", lambda s: None, "SR"),
        "RSVRG (tuned eta, finite sum)": ("rsvrg", lambda s: best.get((s, "rsvrg")), "RN"),
    }
    cfgs = []
    for name, (m, f, mode) in methods.items():
        for s, sc in SCEN.items():
            if mode == "SR" and sc["t"] is None:
                continue
            if m == "rsvrg" and sc["prob"] != "digits":
                continue
            cfgs += [dict(scenario=s, label=name, method=m, eta=f(s), mode=mode, seed=i, K=K,
                          trace=False) for i in test]
    res = pmap(job, cfgs, a.workers)
    rows = []
    for name in methods:
        row = dict(method=name)
        for s in SCEN:
            rs = [r for r in res if r["label"] == name and r["scenario"] == s]
            if not rs:
                row[s] = "NA"
                continue
            m, h = mean_ci([r["err"] for r in rs])
            nd = sum(r["div"] for r in rs)
            row[s] = f"{m:.4f} ± {h:.4f}" + (f" ({nd} div.)" if nd else "")
        rows.append(row)
    write_csv("table6_baselines.csv", rows, ["method"] + list(SCEN))

    # ---- transfer of tuned steps between regimes --------------------------
    tr_cfgs = [dict(scenario="Synthetic N", label="step tuned under noise", method="rsgd",
                    eta=best["Synthetic S", "rsgd"], seed=i, K=K) for i in test]
    tr_cfgs += [dict(scenario="Synthetic S", label="step tuned under precision", method="rsgd",
                     eta=best["Synthetic N", "rsgd"], seed=i, K=K) for i in test]
    res2 = pmap(job, tr_cfgs, a.workers)
    write_csv("table6_transfer.csv",
              [dict(applied_in=s, step=lab, eta=res2[[r["label"] for r in res2].index(lab)]["eta"],
                    error=np.mean([r["err"] for r in res2 if r["label"] == lab]))
               for s, lab in (("Synthetic N", "step tuned under noise"), ("Synthetic S", "step tuned under precision"))])


if __name__ == "__main__":
    main()
