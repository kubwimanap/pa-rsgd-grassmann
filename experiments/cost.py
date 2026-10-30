"""Computational cost of PA-RSGD (Section 3.3).

Measured in double precision on a CPU (simulated rounding would distort the
timings), with minibatch noise and the surrogate retraction defect (S+G):

* time per iteration of each method, relative to fixed-step RSGD (median of
  two runs of 500 iterations);
* iterations and wall-clock time until the error first falls below a target
  (median over 3 seeds, error checked every 10 iterations).

Problems: synthetic (n = 60, p = 4, b = 16, gamma = 0.95, target 0.20) and
spiked (n = 10^4, p = 10, b = 32, gamma = 0.95, target 1.8).

Outputs
  results/tables/cost.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, Retraction, pa_rsgd, random_start, rsgd, spiked, synthetic
from parsgd.util import parse, read_csv, write_csv

PROBS = {
    "synthetic": dict(make=lambda: synthetic(0), b=16, K=4000, target=0.20),
    "spiked": dict(make=lambda: spiked(0), b=32, K=2000, target=1.8),
}


def tuned_step(name):
    if name == "synthetic":
        rows = read_csv("table6_tuned_steps.csv") or []
        hit = [r for r in rows if r["scenario"] == "Synthetic N+S+G" and r["method"] == "rsgd"]
        return float(hit[0]["tuned_eta"]) if hit else 0.003
    rows = read_csv("large_spiked.csv") or []
    hit = [r["method"] for r in rows if r["method"].startswith("RSGD, tuned eta=") and not r["method"].endswith("SR")]
    return float(hit[0].split("=")[1]) if hit else 0.003


def run(P, method, eta_t, b, K, seed, rec):
    rng = np.random.default_rng(seed)
    X0 = random_start(P.n, P.p, rng)
    R = Retraction("surrogate", 0.95)
    ar = Arith()
    t0 = time.perf_counter()
    if method == "RSGD, fixed step":
        tr = rsgd(P, X0, K, 0.02, b, R, ar, rng, rec=rec)
    elif method == "RSGD, decaying step":
        tr = rsgd(P, X0, K, 0.02, b, R, ar, rng, rec=rec, decay=200)
    elif method == "RSGD, tuned step":
        tr = rsgd(P, X0, K, eta_t, b, R, ar, rng, rec=rec)
    elif method == "Landing (tuned step)":
        tr = rsgd(P, X0, K, eta_t, b, Retraction("landing", lam=1.0), ar, rng, rec=rec)
    else:
        tr = pa_rsgd(P, X0, K, b, R, ar, rng)
    return tr, time.perf_counter() - t0


METHODS = ("RSGD, fixed step", "RSGD, decaying step", "RSGD, tuned step", "Landing (tuned step)", "PA-RSGD")


def main():
    a = parse(__doc__)
    S = 2 if a.quick else 3
    rows = []
    for name, spec in PROBS.items():
        P = spec["make"]()
        K = spec["K"] // (4 if a.quick else 1)
        Kt = min(K, 500)
        eta_t = tuned_step(name)
        for m in METHODS:
            # timing: first Kt iterations of two seeds, errors recorded only at the end
            per_it = []
            for s in range(2):
                _, el = run(P, m, eta_t, spec["b"], Kt, 300 + s, rec=Kt)
                per_it.append(el / Kt)
            t_it = float(np.median(per_it))
            hit_it, hit_t, final = [], [], []
            for s in range(S):
                tr, _ = run(P, m, eta_t, spec["b"], K, 300 + s, rec=10)
                err = np.asarray(tr.err)
                it = np.asarray(tr.it)
                below = np.nonzero(err <= spec["target"])[0]
                k = int(it[below[0]]) if len(below) else np.nan
                hit_it.append(k)
                hit_t.append(k * t_it if np.isfinite(k) else np.nan)
                final.append(tr.stationary("err"))
            hit = np.array(hit_it, float)
            rows.append(dict(problem=name, method=m, step=("PA" if m == "PA-RSGD" else eta_t if "tuned" in m else 0.02),
                             us_per_iteration=1e6 * t_it,
                             target=spec["target"], reached=f"{int(np.isfinite(hit).sum())}/{S}",
                             iterations_to_target=np.nanmedian(hit) if np.isfinite(hit).any() else np.nan,
                             seconds_to_target=np.nanmedian(hit_t) if np.isfinite(hit).any() else np.nan,
                             final_error=float(np.mean(final)), seeds=S))
        base = [r for r in rows if r["problem"] == name and r["method"] == "RSGD, fixed step"][0]["us_per_iteration"]
        for r in rows:
            if r["problem"] == name:
                r["relative_cost_per_iteration"] = r["us_per_iteration"] / base
    write_csv("cost.csv", rows, ["problem", "method", "step", "us_per_iteration", "relative_cost_per_iteration",
                                 "target", "reached", "iterations_to_target", "seconds_to_target", "final_error",
                                 "seeds"])


if __name__ == "__main__":
    main()
