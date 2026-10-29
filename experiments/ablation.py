"""Table 5: ablation of PA-RSGD (stationary error, divergences).

Outputs
  results/tables/table5_ablation.csv
  results/tables/table5_precision_always.csv
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import Arith, PAConfig, Retraction, pa_rsgd, random_start, rsgd, synthetic
from parsgd.util import parse, pmap, write_csv

P = synthetic(0)
SCEN = {
    "N+G": dict(t=7, b=None, gamma=0.95, eta0=0.02),
    "S+G": dict(t=None, b=16, gamma=0.95, eta0=0.02),
    "N+S+G": dict(t=7, b=16, gamma=0.95, eta0=0.02),
    "N+S+G, large step": dict(t=7, b=16, gamma=0.99, eta0=0.3),
    "N+S+G, low noise": dict(t=6, b=256, gamma=0.95, eta0=0.02),
}
FULL = PAConfig()
VARIANTS = {
    "RSGD": ("rsgd", {}),
    "RSGD, decaying step": ("rsgd", dict(decay=200)),
    "RSGD + drift re-orth.": ("rsgd", dict(reorth_tau=0.05)),
    "PA step only (no re-orth.)": ("pa", replace(FULL, reorth=False)),
    "PA without precision edge": ("pa", replace(FULL, precision=False)),
    "PA without noise step": ("pa", replace(FULL, noise=False)),
    "PA without upper edge": ("pa", replace(FULL, upper=False)),
    "Full PA-RSGD": ("pa", FULL),
    "PA, precision edge always": ("pa", replace(FULL, precision_always=True)),
}


def job(c):
    sc = SCEN[c["scenario"]]
    rng = np.random.default_rng(7000 + c["seed"])
    X0 = random_start(P.n, P.p, rng)
    ar = Arith(sc["t"], "RN", rng)
    R = Retraction("surrogate", sc["gamma"])
    kind, opt = VARIANTS[c["variant"]]
    if kind == "rsgd":
        tr = rsgd(P, X0, c["K"], sc["eta0"], sc["b"], R, ar, rng, **opt)
    else:
        tr = pa_rsgd(P, X0, c["K"], sc["b"], R, ar, rng, cfg=opt, eta_init=sc["eta0"])
    return dict(c, err=tr.stationary("err"), div=tr.diverged)


def main():
    a = parse(__doc__)
    S = 3 if a.quick else 8
    K = 1500 if a.quick else 4000
    main_variants = [v for v in VARIANTS if v != "PA, precision edge always"]
    cfgs = [dict(scenario=s, variant=v, seed=i, K=K) for v in main_variants for s in SCEN for i in range(S)]
    res = pmap(job, cfgs, a.workers)
    rows = []
    for v in main_variants:
        row = dict(variant=v)
        for s in SCEN:
            rs = [r for r in res if r["variant"] == v and r["scenario"] == s]
            nd = sum(r["div"] for r in rs)
            m = np.mean([r["err"] for r in rs])
            row[s] = f"{m:.4f}" + (f" ({nd}/{len(rs)} div.)" if nd else "")
        rows.append(row)
    write_csv("table5_ablation.csv", rows, ["variant"] + list(SCEN))

    S2 = 2 if a.quick else 4
    cfgs = [dict(scenario="N+S+G, low noise", variant=v, seed=i, K=K)
            for v in ("Full PA-RSGD", "PA, precision edge always") for i in range(S2)]
    res = pmap(job, cfgs, a.workers)
    write_csv("table5_precision_always.csv",
              [dict(variant=v, low_noise_error=np.mean([r["err"] for r in res if r["variant"] == v]), seeds=S2)
               for v in ("Full PA-RSGD", "PA, precision edge always")])


if __name__ == "__main__":
    main()
