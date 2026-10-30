"""Table S5: actual floating-point formats and accumulation.

binary16 and bfloat16, with
  * a wide accumulator (every result rounded to the format),
  * sequential low-precision accumulation (every partial sum rounded),
  * XLA's native reduced-precision matrix kernels through JAX on CPU
    (skipped if JAX is not installed).
Scenarios: N only, and N+S+G (b = 16, gamma = 0.95).

Outputs
  results/tables/tableS5_formats.csv
  results/tables/tableS5_formats_raw.csv   (unformatted means and CIs)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, pa_rsgd, random_start, rsgd, synthetic
from parsgd.arith import NativeArith
from parsgd.util import mean_ci, parse, pmap, write_csv

P = synthetic(0)
SCEN = {"N": dict(b=None, gamma=0.0), "N+S+G": dict(b=16, gamma=0.95)}
METHODS = ("RSGD, fixed eta", "RSGD, decaying eta", "PA-RSGD")


def run(c, ar, rng):
    sc = SCEN[c["scenario"]]
    X0 = random_start(P.n, P.p, rng)
    R = Retraction("surrogate", sc["gamma"])
    if c["method"] == "PA-RSGD":
        return pa_rsgd(P, X0, c["K"], sc["b"], R, ar, rng)
    return rsgd(P, X0, c["K"], 0.02, sc["b"], R, ar, rng, decay=200 if "decaying" in c["method"] else None)


def job(c):
    rng = np.random.default_rng(8000 + c["seed"])
    ar = Arith(fmt=c["fmt"], mode="RN", rng=rng, accumulate=c["acc"])
    return dict(c, err=run(c, ar, rng).stationary("err"))


def main():
    a = parse(__doc__)
    K = 1000 if a.quick else 4000
    S = 2 if a.quick else 10
    cfgs = [dict(fmt=f, acc=acc, scenario=s, method=m, seed=i, K=K)
            for f in ("binary16", "bfloat16") for acc in ("wide", "low") for s in SCEN for m in METHODS
            for i in range(S)]
    res = pmap(job, cfgs, a.workers)
    try:
        import jax  # noqa: F401
        have_jax = True
    except ImportError:
        have_jax = False
    if have_jax:                         # run in this process: JAX is not fork-safe
        S2 = 2 if a.quick else 5
        for f in ("binary16", "bfloat16"):
            for s in SCEN:
                for m in ("RSGD, fixed eta", "PA-RSGD"):
                    for i in range(S2):
                        c = dict(fmt=f, acc="native", scenario=s, method=m, seed=i, K=K)
                        rng = np.random.default_rng(8000 + i)
                        res.append(dict(c, err=run(c, NativeArith(f, rng), rng).stationary("err")))
    rows, raw = [], []
    names = {"wide": "wide accumulator", "low": "low-precision accumulation", "native": "native XLA kernel"}
    for f in ("binary16", "bfloat16"):
        for acc in ("wide", "low", "native"):
            for s in SCEN:
                row = dict(format=f, accumulation=names[acc], scenario=s)
                any_ = False
                for m in METHODS:
                    rr = [r["err"] for r in res if r["fmt"] == f and r["acc"] == acc and r["scenario"] == s
                          and r["method"] == m]
                    if rr:
                        mu, h = mean_ci(rr)
                        row[m] = f"{mu:.4f} ± {h:.4f} ({len(rr)} seeds)"
                        raw.append(dict(format=f, accumulation=names[acc], scenario=s, method=m,
                                        mean=mu, ci=h, seeds=len(rr)))
                        any_ = True
                    else:
                        row[m] = "NA"
                if any_:
                    rows.append(row)
    write_csv("tableS5_formats.csv", rows, ["format", "accumulation", "scenario"] + list(METHODS))
    write_csv("tableS5_formats_raw.csv", raw)


if __name__ == "__main__":
    main()
