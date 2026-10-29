"""Perturbed RSGD, its inexact retractions, PA-RSGD (Algorithm 1) and baselines.

One iteration of the perturbed RSGD (equation (4) of the paper):

    d_k     = fl( (I - X X^T) grad F(X_k; zeta_k) )        (Lemma 3 order)
    Y_k     = fl( X_k - eta_k d_k )
    X_{k+1} = fl( R(Y_k) )

with R one of the inexact retractions below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arith import Arith
from .problems import orth


# --------------------------------------------------------------------------
# retractions
# --------------------------------------------------------------------------
@dataclass
class Retraction:
    """kind: 'surrogate' (gamma), 'none', 'periodic' (T), 'ns' (one Newton-Schulz
    step) or 'landing' (lam).  'surrogate' with gamma = 0 is the exact polar retraction."""

    kind: str = "surrogate"
    gamma: float = 0.0
    T: int = 0
    lam: float = 0.0

    def label(self):
        return {"surrogate": f"surrogate(gamma={self.gamma})", "none": "none",
                "periodic": f"periodic(T={self.T})", "ns": "Newton-Schulz",
                "landing": f"landing(lam={self.lam})"}[self.kind]


def polar(Y):
    return orth(Y)


def apply_retraction(R: Retraction, Y, k, ar: Arith):
    if R.kind == "surrogate":
        if R.gamma == 0.0:
            return ar.fl(polar(Y))
        return ar.fl((1.0 - R.gamma) * polar(Y) + R.gamma * Y)
    if R.kind == "none" or R.kind == "landing":
        return Y
    if R.kind == "periodic":
        return ar.fl(polar(Y)) if (k + 1) % R.T == 0 else Y
    if R.kind == "ns":
        G = ar.matmul(Y.T, Y)
        return ar.matmul(Y, ar.fl((3.0 * np.eye(Y.shape[1]) - G) / 2.0))
    raise ValueError(R.kind)


def direction(problem, X, S, ar: Arith):
    """Projected direction d = (I - X X^T) grad F, in the order of Lemma 3."""
    G = problem.egrad(X, S, ar)
    T = ar.matmul(X.T, G)
    Z = ar.matmul(X, T)
    return ar.fl(G - Z)


def drift(X):
    p = X.shape[1]
    return float(np.linalg.norm(X.T @ X - np.eye(p)))


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class Trace:
    it: list = field(default_factory=list)
    err: list = field(default_factory=list)
    drift: list = field(default_factory=list)
    g2: list = field(default_factory=list)
    eta: list = field(default_factory=list)
    diverged: bool = False

    def stationary(self, key="err", frac=0.2):
        v = np.asarray(getattr(self, key))
        if len(v) == 0:
            return np.nan
        m = max(1, int(np.ceil(frac * len(v))))
        return float(np.mean(v[-m:]))


def _record(tr, problem, X, k, eta, want_g2):
    tr.it.append(k)
    tr.err.append(problem.error(X))
    tr.drift.append(drift(X))
    tr.eta.append(eta)
    if want_g2:
        tr.g2.append(problem.rgrad_norm2(X))


def _diverged(X):
    return (not np.all(np.isfinite(X))) or drift(X) > 1e3


def _mark_divergence(tr, problem):
    tr.diverged = True
    tr.err.append(np.sqrt(problem.p))
    tr.drift.append(np.inf)
    tr.g2.append(np.inf)
    return tr


# --------------------------------------------------------------------------
# fixed-schedule RSGD (also Oja = no retraction, and landing)
# --------------------------------------------------------------------------
def rsgd(problem, X0, K, eta, b, R: Retraction, ar: Arith, rng, rec=20, want_g2=False,
         decay=None, reorth_tau=None):
    """RSGD with step ``eta`` (constant) or ``eta/(1+k/decay)`` if ``decay`` is set."""
    X = ar.fl(X0.copy())
    tr = Trace()
    _record(tr, problem, X, 0, eta, want_g2)
    p = X.shape[1]
    for k in range(K):
        ek = eta / (1.0 + k / decay) if decay else eta
        S = problem.sample(rng, b)
        d = direction(problem, X, S, ar)
        if R.kind == "landing":
            E = X.T @ X - np.eye(p)
            d = ar.fl(d + R.lam * ar.matmul(X, ar.fl(E)))
        Y = ar.fl(X - ar.scale(ek, d))
        X = apply_retraction(R, Y, k, ar)
        if reorth_tau is not None and drift(X) > reorth_tau:
            X = ar.fl(polar(X))
        if _diverged(X):
            return _mark_divergence(tr, problem)
        if (k + 1) % rec == 0:
            _record(tr, problem, X, k + 1, ek, want_g2)
    return tr


# --------------------------------------------------------------------------
# PA-RSGD (Algorithm 1)
# --------------------------------------------------------------------------
@dataclass
class PAConfig:
    tau: float = 0.05          # drift tolerance
    beta: float = 0.98         # averaging factor
    noise: bool = True         # noise step eta_S
    precision: bool = True     # precision edge eta_N (only when stagnation detected)
    precision_always: bool = False
    upper: bool = True         # upper edge eta_max = 1/L_k
    reorth: bool = True        # drift monitor + re-orthonormalisation
    anytime: bool = False      # horizon value with K replaced by k+1
    horizon: bool = True


def pa_rsgd(problem, X0, K, b, R: Retraction, ar: Arith, rng, cfg=PAConfig(), rec=20,
            want_g2=False, eta_init=0.02):
    X = ar.fl(X0.copy())
    n, p = X.shape
    tr = Trace()
    _record(tr, problem, X, 0, eta_init, want_g2)
    u = ar.u
    nk = vk = Lk = None
    phi = 0.0
    for k in range(K):
        S = problem.sample(rng, b)
        d = direction(problem, X, S, ar)
        if S is not None and S.shape[1] >= 2:
            h = S.shape[1] // 2
            d1 = direction(problem, X, S[:, :h], Arith())
            d2 = direction(problem, X, S[:, h:2 * h], Arith())
            vnew = 0.25 * float(np.sum((d1 - d2) ** 2))
        else:
            vnew = 0.0
        nnew = float(np.sum(d * d))
        lnew = float(np.max(np.linalg.eigvalsh(problem.H(X, S))))
        if nk is None:
            nk, vk, Lk = nnew, vnew, max(lnew, 1e-12)
        else:
            nk = cfg.beta * nk + (1 - cfg.beta) * nnew
            vk = cfg.beta * vk + (1 - cfg.beta) * vnew
            Lk = 0.95 * Lk + 0.05 * lnew
        g2 = max(nk - vk, 0.0)
        r = vk / g2 if g2 > 0 else np.inf
        nu = n * u * u
        eta_max = 1.0 / Lk if cfg.upper else np.inf
        cands = []
        if cfg.noise:
            cands.append((1 - 2 * nu) / (3 * Lk * (r + nu)) if np.isfinite(r) else 0.0)
        if cfg.horizon and vk > 0:
            KK = (k + 1) if cfg.anytime else K
            cands.append(np.sqrt(p / (3 * vk * KK)))
        if cfg.precision and (phi >= 0.9 or cfg.precision_always) and u > 0:
            cands.append(u * np.sqrt(p) / np.sqrt(g2 + vk + 1e-300))
        if not cands:
            cands.append(eta_init)
        eta = min(eta_max, max(cands))
        if not np.isfinite(eta):
            eta = eta_init
        Y = ar.fl(X - ar.scale(eta, d))
        frac = float(np.mean(Y == X))
        X = apply_retraction(R, Y, k, ar)
        phi = 0.9 * phi + 0.1 * frac
        if cfg.reorth and drift(X) > cfg.tau:
            X = ar.fl(polar(X))
        if _diverged(X):
            return _mark_divergence(tr, problem)
        if (k + 1) % rec == 0:
            _record(tr, problem, X, k + 1, eta, want_g2)
    return tr


# --------------------------------------------------------------------------
# RSVRG (finite sums only)
# --------------------------------------------------------------------------
def rsvrg(problem, X0, K, eta, b, R: Retraction, ar: Arith, rng, m=50, rec=20, want_g2=False):
    X = ar.fl(X0.copy())
    tr = Trace()
    _record(tr, problem, X, 0, eta, want_g2)
    exact = Arith()
    for k in range(K):
        if k % m == 0:
            Xs = orth(X)
            Gs = problem.egrad(Xs, None, exact)
        S = problem.sample(rng, b)
        G = problem.egrad(X, S, ar) - problem.egrad(Xs, S, ar) + Gs
        G = ar.fl(G)
        T = ar.matmul(X.T, G)
        d = ar.fl(G - ar.matmul(X, T))
        Y = ar.fl(X - ar.scale(eta, d))
        X = apply_retraction(R, Y, k, ar)
        if _diverged(X):
            return _mark_divergence(tr, problem)
        if (k + 1) % rec == 0:
            _record(tr, problem, X, k + 1, eta, want_g2)
    return tr


def random_start(n, p, rng):
    return orth(rng.standard_normal((n, p)))


def near_solution(problem, rng, scale=0.05):
    U = problem.Up
    return orth(U + scale * rng.standard_normal(U.shape))
