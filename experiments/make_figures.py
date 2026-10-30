"""Draw Figures 2-5 and S1 of the article from the CSV files in results/tables.

    python experiments/make_figures.py          # all figures
    python experiments/make_figures.py fig3     # one figure

The experiment scripts call the matching function after writing their CSV
files, so the figures never need a separate simulation run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd.util import read_csv, savefig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10, "axes.titlesize": 10,
    "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
ERR = r"$\|\sin\Theta\|_F$"
C = {"exact": "#222222", "SR": "#1f6fb4", "RN": "#c0392b", "obs": "#1f6fb4", "add": "#e08e0b",
     "dith": "#2a9d55", "b32": "#e08e0b"}


def _f(rows, key):
    return np.array([float(r[key]) for r in rows])


def _slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


# ---------------------------------------------------------------------------
def fig2():
    rows = [r for r in read_csv("fig2_floors.csv") if r["vacuous"] == "False" and float(r["bound"]) > 0]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for key, lab in (("exact", "exact arithmetic"), ("SR", "stochastic rounding"), ("RN", "round-to-nearest")):
        for full, mk in ((False, "o"), (True, "^")):
            rr = [r for r in rows if r["arith"].startswith(key) and (r["b"] == "full") == full]
            if rr:
                ax[0].loglog(_f(rr, "bound"), _f(rr, "observed_g2"), mk, ms=3.2, alpha=0.75, color=C[key],
                             mec="none", label=f"{lab}" + (", full gradient" if full else ""))
    b = _f(rows, "bound")
    lim = np.array([b.min() / 2, b.max() * 2])
    ax[0].loglog(lim, lim, "k--", lw=0.8, label="observed = bound")
    ax[0].set(xlabel=r"a-priori floor $\Phi$ of (18)/(19)", ylabel=r"observed stationary $\|\mathrm{grad}\,f\|_F^2$",
              title="(a)")
    ax[0].legend(loc="upper left")
    groups = (("exact", lambda r: r["arith"] == "exact"),
              ("SR", lambda r: r["arith"].startswith("SR") and r["b"] != "full"),
              ("RN", lambda r: r["arith"].startswith("RN") and r["b"] != "full"),
              ("SR\nfull grad.", lambda r: r["arith"].startswith("SR") and r["b"] == "full"),
              ("RN\nfull grad.", lambda r: r["arith"].startswith("RN") and r["b"] == "full"))
    data = [_f([r for r in rows if sel(r)], "ratio") for _, sel in groups]
    bp = ax[1].boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "k"})
    for patch, col in zip(bp["boxes"], (C["exact"], C["SR"], C["RN"], C["SR"], C["RN"])):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    ax[1].set_xticks(range(1, 6))
    ax[1].set_xticklabels([g for g, _ in groups])
    ax[1].set(yscale="log", ylabel="observed / a-priori floor", title="(b)")
    ax[1].axhline(1, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    savefig(fig, "fig2_floors")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig3():
    rows = read_csv("tableS1_stability.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    gg = np.linspace(0.25, 0.995, 200)
    for tag, col, lab in (("exact", C["exact"], "exact gradient"), ("b32", C["b32"], "$b = 32$")):
        ref = [r for r in rows if float(r["gamma"]) > 0][0]
        g0, e0 = float(ref["gamma"]), float(ref[f"{tag}_theorem1_edge"])
        mu = (1 + 1 / g0) / (2 * e0)
        for r in rows:
            g, lo, hi = float(r["gamma"]), float(r[f"{tag}_last_stable"]), float(r[f"{tag}_first_unstable"])
            ax.errorbar(g, 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt="o", ms=4, color=col, capsize=2)
        ax.plot(gg, (1 + 1 / gg) / (2 * mu), "-", color=col, lw=1.2,
                label=rf"$\eta_G(\gamma)$, {lab} ($\mu_{{\max}}={mu:.2f}$)")
    ax.set(xlabel=r"retraction defect $\gamma$", ylabel=r"largest stable step $\eta$", ylim=(0, 1.3))
    ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig3_stability")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig4():
    fa, fb, fc = read_csv("fig4a_precision_floor.csv"), read_csv("fig4b_stochastic_floor.csv"), \
        read_csv("fig4c_drift_vs_bound.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    x, y = _f(fa, "u_over_eta"), _f(fa, "err")
    ax[0].loglog(x, y, "o", ms=4, color=C["RN"])
    ax[0].set(xlabel=r"$u/\eta$", ylabel="stationary " + ERR, title=f"(a) precision only, RN (slope {_slope(x, y):.2f})")
    x, y = _f(fb, "eta_over_b"), _f(fb, "err2")
    ax[1].loglog(x, y, "o", ms=4, color=C["SR"])
    ax[1].set(xlabel=r"$\eta/b$", ylabel="stationary " + ERR + r"$^2$", title=f"(b) noise only (slope {_slope(x, y):.2f})")
    for t, mk, lab in (("inf", "o", "exact arithmetic"), ("10", "s", "$t = 10$")):
        rr = [r for r in fc if r["t"] == t]
        ax[2].loglog(_f(rr, "bound_D2"), _f(rr, "drift_obs"), mk, ms=4, mfc="none", label=lab)
    x, y = _f(fc, "bound_D2"), _f(fc, "drift_obs")
    lim = [min(x.min(), y.min()) / 2, max(x.max(), y.max()) * 2]
    ax[2].loglog(lim, lim, "k--", lw=0.8)
    ax[2].set(xlabel=r"a-priori bound $D_\infty^{(2)}$", ylabel=r"observed drift $D$", title="(c) drift, Theorem 1(iii)")
    ax[2].legend()
    fig.tight_layout()
    savefig(fig, "fig4_scaling")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig5():
    rows = read_csv("tableS2_additivity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [f"{r['level']}: {r['combination']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    ax[0].bar(x - w, _f(rows, "observed_error"), w, color=C["obs"], label="observed")
    ax[0].bar(x, _f(rows, "additive_prediction"), w, color=C["add"], label="additive prediction")
    ax[0].bar(x + w, _f(rows, "dithering_prediction"), w, color=C["dith"], label="dithering prediction (20)")
    ax[0].set(ylabel="stationary " + ERR, title="(a) error", ylim=(0, 0.68))
    w = 0.38
    ax[1].bar(x - w / 2, np.maximum(_f(rows, "observed_drift"), 1e-16), w, color=C["obs"], label="observed")
    ax[1].bar(x + w / 2, np.maximum(_f(rows, "additive_drift"), 1e-16), w, color=C["add"], label="sum of individual drifts")
    ax[1].set(yscale="log", ylim=(1e-16, 1e4), ylabel=r"drift $D$", title="(b) orthonormality drift")
    for a, nc in zip(ax, (3, 2)):
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=40, ha="right")
        a.legend(loc="upper left", ncol=nc)
    fig.tight_layout()
    savefig(fig, "fig5_additivity")
    plt.close(fig)


# ---------------------------------------------------------------------------
def figS1():
    rows = read_csv("figS1_digits_traces.csv")
    panels = list(dict.fromkeys(r["panel"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, p in enumerate(panels):
        for j, m in enumerate(methods):
            rr = [r for r in rows if r["panel"] == p and r["method"] == m]
            it = _f(rr, "iteration")
            ax[k].plot(it, _f(rr, "median"), color=f"C{j}", lw=1.2, label=m.replace(" eta", r" $\eta$"))
            ax[k].fill_between(it, _f(rr, "q25"), _f(rr, "q75"), color=f"C{j}", alpha=0.18, lw=0)
        ax[k].set(title=p, xlabel="iteration", yscale="log")
    ax[0].set_ylabel(ERR + " (median and interquartile range)")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "figS1_digits")
    plt.close(fig)


ALL = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "figS1": figS1}

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        if read_csv({"fig2": "fig2_floors.csv", "fig3": "tableS1_stability.csv", "fig4": "fig4c_drift_vs_bound.csv",
                     "fig5": "tableS2_additivity.csv", "figS1": "figS1_digits_traces.csv"}[name]) is None:
            print(f"skip {name}: run its experiment first")
            continue
        ALL[name]()
