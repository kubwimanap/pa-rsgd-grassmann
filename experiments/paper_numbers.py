"""Compute every number quoted in the text of the article from results/tables
and write PAPER_NUMBERS.md (one line per quoted statement, with its source).

Run after the experiments.  Also evaluates the sensitivity of the Theorem 3
floors to the predicted drift and to Gamma_N (no new simulations needed).
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from parsgd import spiked, synthetic
from parsgd.bounds import constants_at_solution, floor_bound
from parsgd.util import ROOT, read_csv, write_csv

L = []


def say(section, text):
    L.append((section, text))


def f(rows, key):
    return np.array([float(r[key]) for r in rows])


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def find(rows, **kw):
    for r in rows or []:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    raise KeyError(kw)


def pct(x):
    return f"{100 * x:.1f}%"


ARITH = {"exact": (None, "RN"), "SR t=10": (10, "SR"), "SR t=8": (8, "SR"), "RN t=10": (10, "RN"),
         "RN t=8": (8, "RN")}


def section_31():
    rows = read_csv("fig2_floors.csv")
    nv = [r for r in rows if r["vacuous"] == "False"]
    sec = "3.1 Figure 2"
    say(sec, f"configurations {len(rows)}; exceeded {sum(r['exceeded'] == 'True' for r in rows)}; "
             f"vacuous {sum(r['vacuous'] == 'True' for r in rows)}")
    vac = [r for r in rows if r["vacuous"] == "True"]
    desc = sorted({(r["gap"], r["b"], r["eta"], r["gamma"]) for r in vac})
    say(sec, "vacuous configurations (gap, b, eta, gamma): " + "; ".join(",".join(d) for d in desc))
    say(sec, f"observed drift in vacuous configurations: {f(vac, 'observed_drift').min():.3f}-"
             f"{f(vac, 'observed_drift').max():.3f}; predicted drift {f(vac, 'predicted_drift').min():.2f}-"
             f"{f(vac, 'predicted_drift').max():.2f}")
    for name, sel in (("exact, stochastic", lambda r: r["arith"] == "exact"),
                      ("SR, stochastic", lambda r: r["arith"].startswith("SR") and r["b"] != "full"),
                      ("RN, stochastic", lambda r: r["arith"].startswith("RN") and r["b"] != "full"),
                      ("RN, full gradient", lambda r: r["arith"].startswith("RN") and r["b"] == "full"),
                      ("SR, full gradient", lambda r: r["arith"].startswith("SR") and r["b"] == "full")):
        rr = [r for r in nv if sel(r)]
        q = f(rr, "ratio")
        say(sec, f"{name}: n={len(rr)}, ratio {pct(q.min())}-{pct(q.max())}, median {pct(np.median(q))}, "
                 f"log-log slope vs bound {slope(f(rr, 'bound'), f(rr, 'observed_g2')):.2f}")
    # full-gradient RN: floor vs eta per (gap, t, gamma)
    sl = []
    for gap in sorted({r["gap"] for r in rows}):
        for ar in ("RN t=10", "RN t=8"):
            for g in ("0", "0.9"):
                rr = [r for r in rows if r["gap"] == gap and r["arith"] == ar and r["gamma"] == g and r["b"] == "full"]
                sl.append(slope(f(rr, "eta"), f(rr, "observed_g2")))
    say(sec, f"RN full gradient: floor ~ eta^{np.mean(sl):.2f} (range {min(sl):.2f} to {max(sl):.2f}, "
             f"{len(sl)} combinations)")
    rat = []
    for r in rows:
        if r["b"] == "full" and r["arith"].startswith("RN"):
            t = r["arith"][2:]
            s_ = find(rows, gap=r["gap"], eta=r["eta"], b="full", gamma=r["gamma"], arith="SR" + t)
            rat.append(float(r["observed_g2"]) / float(s_["observed_g2"]))
    say(sec, f"SR lowers full-gradient floors by a factor {min(rat):.0f}-{max(rat):.0f} "
             f"(median {np.median(rat):.0f})")
    # sensitivity of the bound (no new runs)
    tails = {1.0: synthetic(0, tail=(0.2, 1.0)), 2.0: synthetic(0, tail=(0.2, 2.0)), 2.3: synthetic(0, tail=(0.2, 2.3))}
    gap2tail = {f"{P.gap:.3f}": t for t, P in tails.items()}
    consts = {}
    ch = {"drift x2": [], "drift x10": [], "Gamma_N x2": []}
    for r in nv:
        if r["b"] == "full":
            continue
        tl = gap2tail[f"{float(r['gap']):.3f}"]
        eta, b, g = float(r["eta"]), int(r["b"]), float(r["gamma"])
        key = (tl, eta, b)
        if key not in consts:
            consts[key] = constants_at_solution(tails[tl], eta, b)
        C = consts[key]
        t, mode = ARITH[r["arith"]]
        base, D, _ = floor_bound(C, eta, g, t, mode, 4)
        for lab, kw in (("drift x2", dict(Dpred=2 * D)), ("drift x10", dict(Dpred=10 * D))):
            v, _, vac = floor_bound(C, eta, g, t, mode, 4, **kw)
            if np.isfinite(v) and not vac:
                ch[lab].append(v / base - 1)
        if t is not None:
            v, _, _ = floor_bound(replace(C, Gamma_N=2 * C.Gamma_N), eta, g, t, mode, 4)
            ch["Gamma_N x2"].append(v / base - 1)
    for lab, v in ch.items():
        v = np.asarray(v)
        say(sec, f"sensitivity, {lab}: bound changes by median {pct(np.median(v))}, at most {pct(v.max())}")


def section_31_fig3_table3():
    sec = "3.1 Figure 3 / Table 3"
    rows = read_csv("tableS1_stability.csv")
    ins, above = [], []
    for r in rows:
        g = float(r["gamma"])
        if g == 0:
            say(sec, f"gamma=0: exact-gradient boundary in [{float(r['exact_last_stable']):.3f}, "
                     f"{float(r['exact_first_unstable']):.3f}); b=32 boundary [{float(r['b32_last_stable']):.3f}, "
                     f"{float(r['b32_first_unstable']):.3f})")
            continue
        e, lo, hi = float(r["exact_theorem1_edge"]), float(r["exact_last_stable"]), float(r["exact_first_unstable"])
        ins.append(lo <= e <= hi)
        e, lo, hi = float(r["b32_theorem1_edge"]), float(r["b32_last_stable"]), float(r["b32_first_unstable"])
        if g >= 0.7:
            above.append(max(0.0, e / hi - 1))
    say(sec, f"exact gradient: predicted edge inside the bracket for {sum(ins)}/{len(ins)} values gamma>=0.3")
    say(sec, f"b=32, gamma>=0.7: predicted edge at most {pct(max(above))} above the bracket")
    mu = (1 + 1 / 0.9) / (2 * float(find(rows, gamma="0.9")["b32_theorem1_edge"]))
    say(sec, f"mu_max (b=32) = {mu:.2f}")
    P = synthetic(0)
    Lc = P.lam[0] - P.lam[-1]
    say(sec, f"L = lambda_1 - lambda_n = {Lc:.2f}; 1/L = {1 / Lc:.3f}; ratio to gamma=0 boundary "
             f"{float(find(rows, gamma='0')['exact_last_stable']) * Lc:.1f}")
    se = read_csv("scheme_edges.csv")
    for r in se:
        say(sec, f"{r['scheme']}: predicted {float(r['predicted_edge']):.3f}, bracket "
                 f"[{float(r['last_stable']):.3f}, {float(r['first_unstable']):.3f}), inside={r['inside']}")
    t3 = read_csv("table3_practical_retractions.csv")
    rr = [r for r in t3 if "Newton" not in r["scheme"]]
    say(sec, f"observed drift / a-priori bound (all schemes but Newton-Schulz): {pct(f(rr, 'ratio_min').min())}-"
             f"{pct(f(rr, 'ratio_max').max())}")
    ns = find(t3, scheme="one Newton-Schulz step")
    say(sec, f"Newton-Schulz: drift {float(ns['drift_b32']):.1e}-{float(ns['drift_b8']):.1e} (exact arithmetic), "
             f"observed / prediction {pct(float(ns['ratio_min']))}-{pct(float(ns['ratio_max']))}")


def section_32():
    sec = "3.2"
    s3 = read_csv("tableS3_individual.csv")
    g = lambda lv: float(find(s3, level=lv)["err_mean"])  # noqa: E731
    sd = lambda lv: float(find(s3, level=lv)["err_sd"])  # noqa: E731
    say(sec, f"precision floor t=12 {g('t=12'):.4f}, t=7 {g('t=7'):.3f}, t=6 {g('t=6'):.2f} ± {sd('t=6'):.2f}; "
             f"mean factor per bit removed (t=12..7) {np.exp(np.mean(np.diff(np.log([g(f't={t}') for t in range(12, 6, -1)])))):.2f}")
    say(sec, f"noise floor b=4 {g('b=4'):.2f}, b=128 {g('b=128'):.2f}")
    G = [float(r["err_mean"]) for r in s3 if r["perturbation"] == "G"]
    say(sec, f"G alone: error at most {max(G):.1e}")
    fa = read_csv("fig4a_precision_floor.csv")
    P = synthetic(0)
    c = np.median(f(fa, "err") * f(fa, "eta") * P.gap / 2.0 ** -f(fa, "t"))
    say(sec, f"precision floor eps_N ~ {c:.2f} u/(eta delta), log-log slope {slope(f(fa, 'u_over_eta'), f(fa, 'err')):.2f}")
    fb = read_csv("fig4b_stochastic_floor.csv")
    say(sec, f"squared stochastic floor vs eta/b: slope {slope(f(fb, 'eta_over_b'), f(fb, 'err2')):.3f}")
    fc = read_csv("fig4c_drift_vs_bound.csv")
    say(sec, f"drift / D_inf^(2) over {len(fc)} configurations: {f(fc, 'ratio').min():.2f}-{f(fc, 'ratio').max():.2f}")
    sr = read_csv("sr_vs_rn_t7.csv")
    say(sec, f"t=7 no noise: SR {float(find(sr, mode='SR')['err_mean']):.3f}, RN {float(find(sr, mode='RN')['err_mean']):.3f}")
    t4 = read_csv("table4_contrasts.csv")
    for lv in ("A, surrogate", "B, surrogate"):
        e, d = find(t4, level=lv, quantity="error"), find(t4, level=lv, quantity="drift")
        say(sec, f"{lv}: error C_NS {float(e['C_NS']):+.4f}, C_SG {float(e['C_SG']):+.1e}; drift C_NG "
                 f"{float(d['C_NG']):+.4f} [{float(d['C_NG_lo']):+.4f}, {float(d['C_NG_hi']):+.4f}], C_SG "
                 f"{float(d['C_SG']):+.4f}, C_NSG {float(d['C_NSG']):+.4f} [{float(d['C_NSG_lo']):+.4f}, "
                 f"{float(d['C_NSG_hi']):+.4f}], drift C_NS {float(d['C_NS']):+.4f} [{float(d['C_NS_lo']):+.4f}, "
                 f"{float(d['C_NS_hi']):+.4f}]")
    for lv in ("A, periodic (10)", "A, landing (0.5)", "A, Newton-Schulz"):
        e, d = find(t4, level=lv, quantity="error"), find(t4, level=lv, quantity="drift")
        say(sec, f"{lv}: error C_NS {float(e['C_NS']):+.4f}; drift C_SG {float(d['C_SG']):+.2e} "
                 f"[{float(d['C_SG_lo']):+.2e}, {float(d['C_SG_hi']):+.2e}]")
    dv = read_csv("divergence_interaction.csv")
    for r in dv:
        say(sec, f"eta=0.3: {r['case']}: diverged {r['diverged']}, error {float(r['error_finite_runs']):.3f}, "
                 f"drift {float(r['drift_finite_runs']):.2g}")
    for r in read_csv("reconstruction_drift.csv"):
        say(sec, f"reconstruction with X X^T, {r.get('case', r.get('eta'))}: drift {float(r['drift']):.3g}, "
                 f"increase {float(r['reconstruction_increase_pct']):.3g}%")
    s4 = read_csv("tableS4_replication.csv")
    rob = lambda c: float(find(s4, case=c)["robust_error"])  # noqa: E731
    robd = lambda c: float(find(s4, case=c)["robust_drift"])  # noqa: E731
    say(sec, f"robust: N {rob('N'):.3f}, S {rob('S'):.3f}, N+S {rob('N+S'):.3f}, I_err(N+S) "
             f"{rob('N+S') / (rob('N') + rob('S')):.2f}; drift S {robd('S'):.1e}, G {robd('G'):.1e}, S+G {robd('S+G'):.1e}")
    r = find(s4, case="N")
    if not str(r["hopkins_error"]).startswith("not run"):
        hk = lambda c: float(find(s4, case=c)["hopkins_error"])  # noqa: E731
        say(sec, f"Hopkins: N {hk('N'):.3f}, S {hk('S'):.3f}, N+S {hk('N+S'):.3f}, I_err(N+S) "
                 f"{hk('N+S') / (hk('N') + hk('S')):.2f}")
    for r in read_csv("damping_F2.csv"):
        say(sec, f"F2 with C={r['C']}, gamma={r['gamma']}: diverged {r['diverged']}")


def section_33():
    sec = "3.3"
    raw = read_csv("tableS5_formats_raw.csv")
    if raw:
        def m(fmt, acc, sc, meth):
            return float(find(raw, format=fmt, accumulation=acc, scenario=sc, method=meth)["mean"])
        s3 = read_csv("tableS3_individual.csv")
        for fmt, t in (("binary16", 11), ("bfloat16", 8)):
            say(sec, f"{fmt} wide accumulator vs emulated t={t} (fixed-step RSGD, precision only): "
                     f"{m(fmt, 'wide accumulator', 'N', 'RSGD, fixed eta'):.4f} vs "
                     f"{float(find(s3, level=f't={t}')['err_mean']):.4f}")
        diffs = []
        for fmt in ("binary16", "bfloat16"):
            for sc in ("N", "N+S+G"):
                for meth in ("RSGD, fixed eta", "PA-RSGD"):
                    a, b = m(fmt, "native XLA kernel", sc, meth), m(fmt, "wide accumulator", sc, meth)
                    diffs.append(abs(a / b - 1))
        diffs = np.array(diffs)
        say(sec, f"native vs simulated wide accumulator: within 1% in {int((diffs <= 0.01).sum())}/8, max {pct(diffs.max())}")
        for fmt in ("binary16", "bfloat16"):
            a, b = m(fmt, "wide accumulator", "N", "PA-RSGD"), m(fmt, "low-precision accumulation", "N", "PA-RSGD")
            say(sec, f"{fmt} PA-RSGD precision only: wide {a:.4f} -> low-precision accumulation {b:.4f} (x{b / a:.1f})")
            for meth in ("RSGD, fixed eta", "RSGD, decaying eta"):
                a, b = m(fmt, "wide accumulator", "N", meth), m(fmt, "low-precision accumulation", "N", meth)
                say(sec, f"{fmt} {meth} precision only: change {pct(b / a - 1)}")
        ch = []
        for fmt in ("binary16", "bfloat16"):
            for meth in ("RSGD, fixed eta", "RSGD, decaying eta", "PA-RSGD"):
                a, b = m(fmt, "wide accumulator", "N+S+G", meth), m(fmt, "low-precision accumulation", "N+S+G", meth)
                ch.append(abs(b / a - 1))
        say(sec, f"N+S+G: accumulation changes any method by at most {pct(max(ch))}")
        for meth in ("RSGD, fixed eta", "RSGD, decaying eta", "PA-RSGD"):
            a, b = m("binary16", "wide accumulator", "N+S+G", meth), m("bfloat16", "wide accumulator", "N+S+G", meth)
            say(sec, f"N+S+G {meth}: binary16 {a:.4f} vs bfloat16 {b:.4f} ({pct(b / a - 1)})")
    t5 = read_csv("table5_ablation.csv")
    for r in t5:
        say(sec, "Table 5 " + r["variant"] + ": " + "; ".join(f"{k} {v}" for k, v in r.items() if k != "variant"))
    pa = read_csv("table5_precision_always.csv")
    say(sec, "precision edge always on, low noise: " + "; ".join(f"{r['variant']} {float(r['low_noise_error']):.4f}" for r in pa))
    t6 = read_csv("table6_baselines.csv")
    mean = lambda meth, sc: float(find(t6, method=meth)[sc].split()[0])  # noqa: E731
    for sc in ("Synthetic N", "Synthetic S", "Synthetic N+S", "Synthetic N+S+G", "Digits N+S+G"):
        a, b = mean("PA-RSGD", sc), mean("RSGD, oracle-tuned eta", sc)
        say(sec, f"{sc}: PA-RSGD {a:.4f} vs oracle-tuned {b:.4f} ({pct(a / b - 1)}); decaying {mean('RSGD, decaying', sc):.4f}; "
                 f"best method {min(((m['method'], float(m[sc].split()[0])) for m in t6 if m[sc] != 'NA'), key=lambda z: z[1])}")
    tu = read_csv("table6_tuned_steps.csv")
    eN, eS = float(find(tu, scenario="Synthetic N", method="rsgd")["tuned_eta"]), float(find(tu, scenario="Synthetic S", method="rsgd")["tuned_eta"])
    say(sec, f"tuned steps: precision only {eN:.3g}, noise {eS:.3g} (factor {eN / eS:.0f}); digits {float(find(tu, scenario='Digits N+S+G', method='rsgd')['tuned_eta']):.3g}; "
             f"RSVRG {float(find(tu, scenario='Digits N+S+G', method='rsvrg')['tuned_eta']):.3g}")
    for r in read_csv("table6_transfer.csv"):
        say(sec, f"transfer: {r['step']} applied in {r['applied_in']}: {float(r['error']):.2f}")
    sp = read_csv("large_spiked.csv")
    P = spiked(0)
    s02 = np.sum(P.lam[:P.p]) * np.sum(P.lam[P.p:]) / 32
    say(sec, f"spiked: sigma_0^2 = {s02:.3g}; max error sqrt(p) = {np.sqrt(P.p):.2f}")
    for r in sp:
        say(sec, f"spiked {r['method']}: error {float(r['error_mean']):.3f} ± {float(r['error_ci']):.3f}, drift {float(r['drift_mean']):.3g}")
    hm = read_csv("hopkins_methods.csv")
    for r in hm or []:
        say(sec, f"Hopkins N+S+G {r['method']}: {float(r['error_mean']):.3f} ± {float(r['error_sd']):.3f} (sd over problems and seeds)")


def main():
    section_31()
    section_31_fig3_table3()
    section_32()
    section_33()
    out = ["# Numbers quoted in the text", "",
           "Generated by `experiments/paper_numbers.py` from the CSV files in `results/tables`.", ""]
    cur = None
    for sec, text in L:
        if sec != cur:
            out += ["", f"## Section {sec}", ""]
            cur = sec
        out.append(f"- {text}")
    open(os.path.join(ROOT, "PAPER_NUMBERS.md"), "w").write("\n".join(out) + "\n")
    write_csv("paper_numbers.csv", [dict(section=s, statement=t) for s, t in L])
    print("wrote PAPER_NUMBERS.md")


if __name__ == "__main__":
    main()
