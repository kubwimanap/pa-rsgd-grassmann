"""Small helpers shared by the experiment scripts."""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")


def parse(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="small run (fewer seeds/iterations) for testing")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return ap.parse_args()


def pmap(fn, items, workers):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (8 * workers))))


def write_csv(name, rows, fields=None):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    if not rows:
        return path
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    print("wrote", os.path.relpath(path, ROOT))
    return path


def read_csv(name):
    path = os.path.join(TABLES, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return v


def savefig(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", f"results/figures/{name}.png/.pdf")


def mean_ci(x):
    """Mean and 95% confidence half-width (normal approximation)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
