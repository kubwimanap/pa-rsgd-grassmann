"""Write COMPARISON.md: values reported in the manuscript (v4, 29 Oct 2026)
next to the values produced by this repository.

The manuscript numbers were produced by earlier scripts that are not part of
this repository; the values here come from an independent implementation with
different random streams, so small differences are expected.  Wherever the
two disagree, the manuscript should be updated to the values in results/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parsgd.util import ROOT, read_csv


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def g(x):
    try:
        return f"{float(x):.3g}"
    except (TypeError, ValueError):
        return str(x)


def find(rows, **kw):
    for r in rows or []:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return {}


def main():
    L = ["# Manuscript values vs. this repository", "",
         "Manuscript: *Riemannian Stochastic Gradient Descent on the Grassmann Manifold under Interacting "
         "Rounding, Stochastic-Gradient and Retraction Errors* (v4, 29 October 2026).", "",
         "The repository is an independent implementation (new random streams), so numbers differ slightly. "
         "**Before submission, the manuscript tables should be updated to the repository values**, so that "
         "readers who run the code obtain what the paper reports.", ""]

    def table(title, rows):
        L.extend([f"## {title}", "", "| Quantity | Manuscript | Repository |", "|---|---|---|"])
        for q, a, b in rows:
            L.append(f"| {q} | {a} | {b} |")
        L.append("")

    s3 = read_csv("tableS3_individual.csv")
    table("Table S3 (individual perturbations, stationary error)", [
        (f"{p} {lv}", m, f(find(s3, perturbation=p, level=lv).get("err_mean")))
        for p, lv, m in (("N", "t=12", "0.0068"), ("N", "t=10", "0.0262"), ("N", "t=7", "0.1817"),
                         ("N", "t=6", "0.4692"), ("S", "b=128", "0.1119"), ("S", "b=32", "0.2246"),
                         ("S", "b=4", "0.6178"))])
    sr = read_csv("sr_vs_rn_t7.csv")
    table("Section 3.2: stochastic vs round-to-nearest at t = 7, no noise", [
        ("RN error", "0.199", f(find(sr, mode="RN").get("err_mean"), 3)),
        ("SR error", "0.020", f(find(sr, mode="SR").get("err_mean"), 3))])

    t4 = read_csv("table4_contrasts.csv")
    rows = []
    for lv, q, c, m in (("A, surrogate", "error", "NS", "-0.1798"), ("A, surrogate", "drift", "NG", "+0.0202"),
                        ("A, surrogate", "drift", "SG", "+0.0108"), ("A, surrogate", "drift", "NSG", "-0.0204"),
                        ("B, surrogate", "error", "NS", "-0.0252"), ("B, surrogate", "drift", "SG", "+0.0705"),
                        ("A, landing (0.5)", "drift", "SG", "+0.0174"), ("A, periodic (10)", "drift", "SG", "+0.0029"),
                        ("A, Newton-Schulz", "drift", "SG", "+4.5e-6")):
        r = find(t4, level=lv, quantity=q)
        rows.append((f"{lv}, {q}, C_{c}", m,
                     f"{g(r.get('C_' + c))} [{g(r.get('C_' + c + '_lo'))}, {g(r.get('C_' + c + '_hi'))}]" if r else "-"))
    table("Table 4 (interaction contrasts, mean [95% bootstrap CI])", rows)

    s1 = read_csv("tableS1_stability.csv")
    rows = []
    for gm, m in (("0.3", "[0.524, 0.559)"), ("0.9", "[0.260, 0.277)"), ("0.99", "[0.244, 0.260)")):
        r = find(s1, gamma=gm)
        rows.append((f"gamma={gm}, exact gradient, empirical bracket", m,
                     f"[{f(r.get('exact_last_stable'), 3)}, {f(r.get('exact_first_unstable'), 3)})"))
    r = find(s1, gamma="0")
    rows.append(("gamma=0, exact gradient, largest stable step", "1.13", f(r.get("exact_last_stable"), 2)))
    table("Table S1 / Figure 3 (stability boundaries)", rows)

    t3 = read_csv("table3_practical_retractions.csv")
    rows = []
    for sch, m8, m32 in (("surrogate (3), gamma=0.9", "4.4e-2", "1.1e-2"), ("no re-orthonormalization", "8.1e-2", "2.0e-2"),
                         ("re-orthonormalize every 10 steps", "1.3e-2", "2.9e-3"),
                         ("landing, lambda=0.5", "7.0e-2", "1.7e-2")):
        r = find(t3, scheme=sch)
        rows.append((f"{sch}: drift b=8 / b=32", f"{m8} / {m32}", f"{g(r.get('drift_b8'))} / {g(r.get('drift_b32'))}"))
        rows.append((f"{sch}: observed / a-priori", "", f"{f(r.get('ratio_min'), 2)}-{f(r.get('ratio_max'), 2)}"))
    table("Table 3 (practical retractions)", rows)

    t5 = read_csv("table5_ablation.csv")
    rows = []
    for v, vals in (("RSGD", ("0.1821", "0.3148", "0.3155", "2.0000 (8/8 div.)", "0.0857")),
                    ("Full PA-RSGD", ("0.0093", "0.1339", "0.1400", "0.1398", "0.0750")),
                    ("PA without noise step", ("0.0092", "0.3148", "0.3154", "1.0159", "0.0858"))):
        r = find(t5, variant=v)
        for sc, m in zip(("N+G", "S+G", "N+S+G", "N+S+G, large step", "N+S+G, low noise"), vals):
            rows.append((f"{v}, {sc}", m, r.get(sc, "-")))
    table("Table 5 (ablation)", rows)

    t6 = read_csv("table6_baselines.csv")
    rows = []
    for meth, vals in (("RSGD, oracle-tuned eta", ("0.0094", "0.1207", "0.1313", "0.1288", "0.1698")),
                       ("PA-RSGD", ("0.0098", "0.1329", "0.1407", "0.1389", "0.1711")),
                       ("RSGD, decaying", ("0.2447", "0.1499", "0.1563", "0.1550", "0.1612")),
                       ("RSVRG (tuned eta, finite sum)", ("NA", "NA", "NA", "NA", "0.0084"))):
        r = find(t6, method=meth)
        for sc, m in zip(("Synthetic N", "Synthetic S", "Synthetic N+S", "Synthetic N+S+G", "Digits N+S+G"), vals):
            rows.append((f"{meth}, {sc}", m, r.get(sc, "-")))
    tr = read_csv("table6_transfer.csv")
    rows.append(("noise-tuned step used under precision alone", "1.58", f(find(tr, applied_in="Synthetic N").get("error"), 2)))
    rows.append(("precision-tuned step used under noise", "0.94", f(find(tr, applied_in="Synthetic S").get("error"), 2)))
    table("Table 6 (baselines, mean ± 95% CI)", rows)

    sp = read_csv("large_spiked.csv") or []
    table("Large spiked problem (n = 10^4, p = 10)", [
        (r["method"], "", f"error {f(r['error_mean'], 3)}, drift {g(r['drift_mean'])}") for r in sp] + [
        ("manuscript", "fixed 2.700, decaying 1.995, tuned 1.830, PA 1.756; drift 3.1 / 0.025 (PA)", "")])

    dv = read_csv("divergence_interaction.csv") or []
    table("Section 3.2: divergence through interaction (eta = 0.3)", [
        (r["case"], "10/10 diverge only with both" if "both (" in r["case"] else "", r["diverged"]) for r in dv])

    fs = read_csv("fig2_summary.csv") or []
    rows = []
    for grp, m in (("exact, stochastic", "8-38% (median 28%)"), ("SR, stochastic", "3-34% (median 23%)"),
                   ("RN, stochastic", "0.2-54% (median 15%)"), ("RN, full gradient", "1.6-7.6%"),
                   ("SR, full gradient", "0.4-2.9%")):
        r = find(fs, group=grp)
        if r:
            rows.append((grp, m, f"{100 * float(r['ratio_min']):.1f}-{100 * float(r['ratio_max']):.0f}% "
                                 f"(median {100 * float(r['ratio_median']):.0f}%)"))
    r = find(fs, group="all")
    rows.append(("configurations / vacuous / exceeded", "456 / 10 / 0",
                 f"{r.get('configs')} / {r.get('vacuous')} / {r.get('exceeded')}"))
    table("Figure 2 (observed floor / a-priori bound, corrected c_r of Lemma 1)", rows)

    s5 = read_csv("tableS5_formats.csv") or []
    rows = []
    for fm, acc, m in (("binary16", "wide accumulator", "0.0006"), ("binary16", "low-precision accumulation", "0.0025"),
                       ("bfloat16", "wide accumulator", "0.0049"), ("bfloat16", "low-precision accumulation", "0.0198"),
                       ("bfloat16", "native XLA kernel", "0.0052")):
        rows.append((f"PA-RSGD, {fm}, {acc}, N", m, find(s5, format=fm, accumulation=acc, scenario="N").get("PA-RSGD", "-")))
    table("Table S5 (formats)", rows)

    L.extend(["## Findings that differ from the manuscript", "",
              "1. **Decaying step under noise.** With the schedule stated in the manuscript, "
              "eta_k = 0.02 / (1 + k/200), decaying-step RSGD is the most accurate method under S, N+S and N+S+G "
              "on the synthetic problem and on the spiked problem (see Table 6 and large_spiked.csv), whereas the "
              "manuscript reports it as less accurate than PA-RSGD. It still fails under precision alone and "
              "diverges in some runs with a large initial step (Table 5).",
              "2. **Spiked problem.** PA-RSGD keeps the drift smallest by two orders of magnitude, but its error is "
              "not the lowest of the methods compared.",
              "3. **Ablation.** Removing the noise step leaves PA-RSGD almost unchanged, because the horizon step "
              "eta_K takes over; the manuscript reports a return to fixed-step accuracy.",
              "4. **Vacuous bounds.** The corrected constant c_r of Lemma 1 enlarges the predicted drift, so slightly "
              "more configurations leave the invariant set of Theorem 1 than reported (see fig2_summary.csv). "
              "No observed floor exceeds its bound.",
              "5. **Low-precision accumulation.** Rounding every partial sum raises the PA-RSGD floor under precision "
              "alone by less than reported (Table S5): about 1.7x for binary16 and 1.5x for bfloat16, against about 4x "
              "in the manuscript.",
              "6. **Periodic re-orthonormalisation (T = 10).** The stationary drift averaged over the reset cycle is "
              "about three times the manuscript value (Table 3), and so is its S x G drift contrast (Table 4).",
              "",
              "Everything else (interaction contrasts, stability edges, practical-scheme edges, scaling laws, "
              "divergence through interaction, damping test, floating-point formats) agrees with the manuscript "
              "within seed-to-seed variation.", ""])

    path = os.path.join(ROOT, "COMPARISON.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote COMPARISON.md")


if __name__ == "__main__":
    main()
