"""A-priori constants and bounds of Theorems 1 and 3 (Rayleigh quotient).

All constants are fixed in advance from the problem and the data model,
evaluated at the solution U_p; nothing is taken from the runs.

Uses the corrected rounding constant of Lemma 1,
    c_r = 2 sqrt(p) (1 + gamma kappa_star) (1 + O(u)),
since the rounding of Y_k is passed through the retraction and the rounding
of X_{k+1} is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Constants:
    L: float
    sigma0_2: float
    Gamma_N: float
    m2: float
    m4_half: float
    q1: float
    q2: float
    qstar: float
    mu: float
    mu_min: float


def constants_at_solution(problem, eta, b, n_mc=400, seed=12345):
    """Monte Carlo over the data model at U_p."""
    rng = np.random.default_rng(seed)
    U = problem.Up
    lam = problem.lam
    p = problem.p
    L = float(lam[0] - lam[-1])
    if b is None:
        AU = problem.AX(U)
        H = U.T @ AU
        mu = np.linalg.eigvalsh(0.5 * (H + H.T))
        q = np.max(np.abs(1 - eta * (mu[:, None] + mu[None, :])))
        return Constants(L, 0.0, 5 * np.linalg.norm(AU), 0.0, 0.0, q, q, q, mu.max(), mu.min())
    sigma0_2 = float(np.sum(lam[:p]) * np.sum(lam[p:]) / b)
    G2, d2, d4, qs, mus, mumins = [], [], [], [], [], []
    for _ in range(n_mc):
        S = problem.sample(rng, b)
        AkU = S @ (S.T @ U) / b
        Gam = 5 * np.linalg.norm(AkU) + 2 * np.linalg.norm(S, 2) * np.linalg.norm(S.T @ U) / b
        G2.append(Gam ** 2)
        d = AkU - U @ (U.T @ AkU)
        nd = float(np.sum(d * d))
        d2.append(nd)
        d4.append(nd ** 2)
        H = U.T @ AkU
        mu = np.linalg.eigvalsh(0.5 * (H + H.T))
        qs.append(np.max(np.abs(1 - eta * (mu[:, None] + mu[None, :]))))
        mus.append(mu.max())
        mumins.append(mu.min())
    qs = np.asarray(qs)
    return Constants(L, sigma0_2, float(np.sqrt(np.mean(G2))), float(np.mean(d2)),
                     float(np.sqrt(np.mean(d4))), float(np.mean(qs)),
                     float(np.sqrt(np.mean(qs ** 2))), float(np.max(qs)),
                     float(np.max(mus)), float(np.min(mumins)))


def c_r(p, gamma, kappa=1.0):
    return 2.0 * np.sqrt(p) * (1.0 + gamma * kappa)


def drift_bounds(C: Constants, eta, gamma, u, p, kappa=1.0):
    """Theorem 1 (ii)-(iii): first-moment and second-moment drift bounds."""
    cru = c_r(p, gamma, kappa) * u
    rho1, rho2 = gamma * kappa * C.q1, gamma * kappa * C.q2
    D1 = (gamma * kappa * eta ** 2 * C.m2 + cru) / (1 - rho1) if rho1 < 1 else np.inf
    D2 = (gamma * kappa * eta ** 2 * C.m4_half + cru) / (1 - rho2) if rho2 < 1 else np.inf
    return D1, D2


def drift_prediction_scheme(C: Constants, eta, scheme, u, p):
    """A-priori drift for the practical schemes of Corollary 1 (second moment)."""
    kind = scheme.kind
    if kind == "surrogate":
        return drift_bounds(C, eta, scheme.gamma, u, p)[1]
    if kind in ("none", "periodic"):
        D = drift_bounds(C, eta, 1.0, u, p)[1]
        if kind == "none":
            return D
        rho = C.q2
        j = np.arange(1, scheme.T + 1)
        return float(np.mean(D * (1 - rho ** j)))        # averaged over the reset cycle
    if kind == "landing":
        return (eta ** 2 * C.m4_half + c_r(p, 1.0) * u) / (2 * eta * (max(C.mu_min, 0) + scheme.lam))
    if kind == "ns":
        return 0.75 * eta ** 4 * C.m4_half ** 2 + c_r(p, 0.0) * u
    raise ValueError(kind)


def floor_bound(C: Constants, eta, gamma, t, mode, p, Dpred=None):
    """Stationary floor Phi of Theorem 3 ((18) for SR/exact, (19) for RN).

    Returns (Phi, D, vacuous) where vacuous flags that the predicted drift
    leaves the invariant set of Theorem 1 or that theta <= 0.
    """
    if t is None:
        u = 0.0
    else:
        u = 2.0 ** (-t) if mode == "RN" else 2.0 ** (1 - t)
    D = drift_bounds(C, eta, gamma, u, p)[1] if Dpred is None else Dpred
    # invariant set (13) requires D <= theta with sigma_min(X) >= sqrt(1 - theta) >= 1/2,
    # i.e. theta <= 3/4 (the eta M margin of Theorem 1 is neglected here)
    vac = (not np.isfinite(D)) or D > 0.75
    s0 = np.sqrt(C.sigma0_2)
    x = eta * D * C.mu
    beta = x / (1 - x) if x < 1 else np.inf
    if not np.isfinite(beta):
        return np.inf, D, True
    e = u * C.Gamma_N + beta * s0
    L = C.L
    if t is None or mode == "SR":
        if L * eta > 1:
            return np.inf, D, True
        wsr2 = u * u * p / 2.0
        Phi = (1 - L * eta) * (beta * s0) ** 2 + L * eta * (s0 + e) ** 2 + L * wsr2 / eta
        return float(Phi), D, vac
    # round-to-nearest: minimise over the Young parameters a, b, c (epsilon in closed form)
    w2 = (2 * u * np.sqrt(p)) ** 2

    def phi(z):
        a, bb, c = np.exp(z)
        theta = 2 * (1 - c / 2 - bb) - L * eta * (1 + a)
        if theta <= 1e-9 or L * eta * (1 + a) > 2:
            return 1e30
        val = e * e / c + L * eta * (1 + a) * (s0 + e) ** 2 + w2 / (2 * bb * eta ** 2) \
            + L * (1 + 1 / a) * w2 / eta
        return val / theta

    best = np.inf
    for z0 in ([0.0, -2.0, -2.0], [-1.0, -1.0, -3.0], [1.0, -3.0, -1.0]):
        r = minimize(phi, np.asarray(z0), method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 4000})
        best = min(best, r.fun)
    return float(best), D, vac or best >= 1e29


def surrogate_edge(gamma, mu_max, kappa=1.0):
    """Drift-stability edge (14): eta_G(gamma) = (1 + 1/(gamma kappa)) / (2 mu_max)."""
    return np.inf if gamma == 0 else (1 + 1 / (gamma * kappa)) / (2 * mu_max)
