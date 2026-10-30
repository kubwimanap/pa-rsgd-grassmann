# Manuscript values vs. this repository

Manuscript: *Riemannian Stochastic Gradient Descent on the Grassmann Manifold under Interacting Rounding, Stochastic-Gradient and Retraction Errors* (v4, 29 October 2026).

The repository is an independent implementation (new random streams), so numbers differ slightly. **Before submission, the manuscript tables should be updated to the repository values**, so that readers who run the code obtain what the paper reports.

## Table S3 (individual perturbations, stationary error)

| Quantity | Manuscript | Repository |
|---|---|---|
| N t=12 | 0.0068 | 0.0076 |
| N t=10 | 0.0262 | 0.0288 |
| N t=7 | 0.1817 | 0.2019 |
| N t=6 | 0.4692 | 0.3924 |
| S b=128 | 0.1119 | 0.1196 |
| S b=32 | 0.2246 | 0.2381 |
| S b=4 | 0.6178 | 0.6536 |

## Section 3.2: stochastic vs round-to-nearest at t = 7, no noise

| Quantity | Manuscript | Repository |
|---|---|---|
| RN error | 0.199 | 0.202 |
| SR error | 0.020 | 0.021 |

## Table 4 (interaction contrasts, mean [95% bootstrap CI])

| Quantity | Manuscript | Repository |
|---|---|---|
| A, surrogate, error, C_NS | -0.1798 | -0.193 [-0.198, -0.189] |
| A, surrogate, drift, C_NG | +0.0202 | 0.0265 [0.0209, 0.0347] |
| A, surrogate, drift, C_SG | +0.0108 | 0.0119 [0.0119, 0.0119] |
| A, surrogate, drift, C_NSG | -0.0204 | -0.0262 [-0.0344, -0.0207] |
| B, surrogate, error, C_NS | -0.0252 | -0.0279 [-0.0284, -0.0274] |
| B, surrogate, drift, C_SG | +0.0705 | 0.0773 [0.077, 0.0776] |
| A, landing (0.5), drift, C_SG | +0.0174 | 0.0191 [0.0191, 0.0192] |
| A, periodic (10), drift, C_SG | +0.0029 | 0.00942 [0.00938, 0.00946] |
| A, Newton-Schulz, drift, C_SG | +4.5e-6 | 5.24e-06 [5.03e-06, 5.44e-06] |

## Table S1 / Figure 3 (stability boundaries)

| Quantity | Manuscript | Repository |
|---|---|---|
| gamma=0.3, exact gradient, empirical bracket | [0.524, 0.559) | [0.524, 0.559) |
| gamma=0.9, exact gradient, empirical bracket | [0.260, 0.277) | [0.260, 0.277) |
| gamma=0.99, exact gradient, empirical bracket | [0.244, 0.260) | [0.244, 0.260) |
| gamma=0, exact gradient, largest stable step | 1.13 | 1.13 |

## Table 3 (practical retractions)

| Quantity | Manuscript | Repository |
|---|---|---|
| surrogate (3), gamma=0.9: drift b=8 / b=32 | 4.4e-2 / 1.1e-2 | 0.0484 / 0.0119 |
| surrogate (3), gamma=0.9: observed / a-priori |  | 0.20-0.38 |
| no re-orthonormalization: drift b=8 / b=32 | 8.1e-2 / 2.0e-2 | 0.0899 / 0.0222 |
| no re-orthonormalization: observed / a-priori |  | 0.08-0.27 |
| re-orthonormalize every 10 steps: drift b=8 / b=32 | 1.3e-2 / 2.9e-3 | 0.0391 / 0.00948 |
| re-orthonormalize every 10 steps: observed / a-priori |  | 0.16-0.38 |
| landing, lambda=0.5: drift b=8 / b=32 | 7.0e-2 / 1.7e-2 | 0.0774 / 0.0191 |
| landing, lambda=0.5: observed / a-priori |  | 0.05-0.16 |

## Table 5 (ablation)

| Quantity | Manuscript | Repository |
|---|---|---|
| RSGD, N+G | 0.1821 | 0.1919 |
| RSGD, S+G | 0.3148 | 0.3370 |
| RSGD, N+S+G | 0.3155 | 0.3379 |
| RSGD, N+S+G, large step | 2.0000 (8/8 div.) | 1.9683 (8/8 div.) |
| RSGD, N+S+G, low noise | 0.0857 | 0.0902 |
| Full PA-RSGD, N+G | 0.0093 | 0.0094 |
| Full PA-RSGD, S+G | 0.1339 | 0.1395 |
| Full PA-RSGD, N+S+G | 0.1400 | 0.1463 |
| Full PA-RSGD, N+S+G, large step | 0.1398 | 0.1465 |
| Full PA-RSGD, N+S+G, low noise | 0.0750 | 0.0771 |
| PA without noise step, N+G | 0.0092 | 0.0098 |
| PA without noise step, S+G | 0.3148 | 0.1395 |
| PA without noise step, N+S+G | 0.3154 | 0.1456 |
| PA without noise step, N+S+G, large step | 1.0159 | 0.1454 |
| PA without noise step, N+S+G, low noise | 0.0858 | 0.0782 |

## Table 6 (baselines, mean ± 95% CI)

| Quantity | Manuscript | Repository |
|---|---|---|
| RSGD, oracle-tuned eta, Synthetic N | 0.0094 | 0.0097 ± 0.0007 |
| RSGD, oracle-tuned eta, Synthetic S | 0.1207 | 0.1285 ± 0.0023 |
| RSGD, oracle-tuned eta, Synthetic N+S | 0.1313 | 0.1387 ± 0.0026 |
| RSGD, oracle-tuned eta, Synthetic N+S+G | 0.1288 | 0.1365 ± 0.0029 |
| RSGD, oracle-tuned eta, Digits N+S+G | 0.1698 | 0.1913 ± 0.0554 |
| PA-RSGD, Synthetic N | 0.0098 | 0.0100 ± 0.0003 |
| PA-RSGD, Synthetic S | 0.1329 | 0.1383 ± 0.0022 |
| PA-RSGD, Synthetic N+S | 0.1407 | 0.1466 ± 0.0018 |
| PA-RSGD, Synthetic N+S+G | 0.1389 | 0.1441 ± 0.0021 |
| PA-RSGD, Digits N+S+G | 0.1711 | 0.2348 ± 0.1540 |
| RSGD, decaying, Synthetic N | 0.2447 | 0.3691 ± 0.1162 |
| RSGD, decaying, Synthetic S | 0.1499 | 0.0779 ± 0.0024 |
| RSGD, decaying, Synthetic N+S | 0.1563 | 0.1043 ± 0.0029 |
| RSGD, decaying, Synthetic N+S+G | 0.1550 | 0.0951 ± 0.0013 |
| RSGD, decaying, Digits N+S+G | 0.1612 | 0.2071 ± 0.0799 |
| RSVRG (tuned eta, finite sum), Synthetic N | NA | NA |
| RSVRG (tuned eta, finite sum), Synthetic S | NA | NA |
| RSVRG (tuned eta, finite sum), Synthetic N+S | NA | NA |
| RSVRG (tuned eta, finite sum), Synthetic N+S+G | NA | NA |
| RSVRG (tuned eta, finite sum), Digits N+S+G | 0.0084 | 0.0074 ± 0.0002 |
| noise-tuned step used under precision alone | 1.58 | 1.66 |
| precision-tuned step used under noise | 0.94 | 0.98 |

## Large spiked problem (n = 10^4, p = 10)

| Quantity | Manuscript | Repository |
|---|---|---|
| RSGD, eta=0.02 |  | error 2.699, drift 3.12 |
| RSGD, decaying |  | error 1.300, drift 0.281 |
| RSGD, tuned eta=0.003 |  | error 1.506, drift 0.313 |
| RSGD, tuned eta=0.003, SR |  | error 1.509, drift 0.179 |
| PA-RSGD |  | error 1.740, drift 0.0242 |
| manuscript | fixed 2.700, decaying 1.995, tuned 1.830, PA 1.756; drift 3.1 / 0.025 (PA) |  |

## Section 3.2: divergence through interaction (eta = 0.3)

| Quantity | Manuscript | Repository |
|---|---|---|
| noise only (gamma=0, b=16) |  | 0/10 |
| defect only (gamma=0.99, exact gradient) |  | 0/10 |
| both (gamma=0.99, b=16) | 10/10 diverge only with both | 10/10 |
| both, PA-RSGD |  | 0/10 |

## Figure 2 (observed floor / a-priori bound, corrected c_r of Lemma 1)

| Quantity | Manuscript | Repository |
|---|---|---|
| exact, stochastic | 8-38% (median 28%) | 11.9-37% (median 25%) |
| SR, stochastic | 3-34% (median 23%) | 8.7-33% (median 21%) |
| RN, stochastic | 0.2-54% (median 15%) | 0.2-52% (median 13%) |
| RN, full gradient | 1.6-7.6% | 1.6-7% (median 5%) |
| SR, full gradient | 0.4-2.9% | 0.3-3% (median 1%) |
| configurations / vacuous / exceeded | 456 / 10 / 0 | 456 / 17 / 0 |

## Table S5 (formats)

| Quantity | Manuscript | Repository |
|---|---|---|
| PA-RSGD, binary16, wide accumulator, N | 0.0006 | 0.0006 ± 0.0000 (10 seeds) |
| PA-RSGD, binary16, low-precision accumulation, N | 0.0025 | 0.0010 ± 0.0000 (10 seeds) |
| PA-RSGD, bfloat16, wide accumulator, N | 0.0049 | 0.0051 ± 0.0002 (10 seeds) |
| PA-RSGD, bfloat16, low-precision accumulation, N | 0.0198 | 0.0079 ± 0.0002 (10 seeds) |
| PA-RSGD, bfloat16, native XLA kernel, N | 0.0052 | 0.0050 ± 0.0002 (5 seeds) |

## Findings that differ from the manuscript

1. **Decaying step under noise.** With the schedule stated in the manuscript, eta_k = 0.02 / (1 + k/200), decaying-step RSGD is the most accurate method under S, N+S and N+S+G on the synthetic problem and on the spiked problem (see Table 6 and large_spiked.csv), whereas the manuscript reports it as less accurate than PA-RSGD. It still fails under precision alone and diverges in some runs with a large initial step (Table 5).
2. **Spiked problem.** PA-RSGD keeps the drift smallest by two orders of magnitude, but its error is not the lowest of the methods compared.
3. **Ablation.** Removing the noise step leaves PA-RSGD almost unchanged, because the horizon step eta_K takes over; the manuscript reports a return to fixed-step accuracy.
4. **Vacuous bounds.** The corrected constant c_r of Lemma 1 enlarges the predicted drift, so slightly more configurations leave the invariant set of Theorem 1 than reported (see fig2_summary.csv). No observed floor exceeds its bound.
5. **Low-precision accumulation.** Rounding every partial sum raises the PA-RSGD floor under precision alone by less than reported (Table S5): about 1.7x for binary16 and 1.5x for bfloat16, against about 4x in the manuscript.
6. **Hopkins 155.** The draft used per-motion problems with p = 3; for the sequence 1R2RC these are ill-conditioned (gap below 0.01 after scaling), so the repository uses centred trajectories with p = 2 (gaps 0.9-2.5) and, so far, the single sequence 1R2RC.
7. **Periodic re-orthonormalisation (T = 10).** The stationary drift averaged over the reset cycle is about three times the manuscript value (Table 3), and so is its S x G drift contrast (Table 4).

Everything else (interaction contrasts, stability edges, practical-scheme edges, scaling laws, divergence through interaction, damping test, floating-point formats) agrees with the manuscript within seed-to-seed variation.

## Table S4 (Hopkins subset, stationary error; different problem set, see note 6)

| Quantity | Manuscript | Repository |
|---|---|---|
| N | 0.4927 | 0.1887 |
| S | 0.0209 | 0.0055 |
| N+S | 0.3189 | 0.0517 |

