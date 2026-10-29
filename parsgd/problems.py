"""Test problems of the paper.

Every problem exposes

* ``n, p``                      ambient dimension and subspace dimension
* ``sample(rng, b)``            a minibatch S (n x b), or ``None`` for the exact gradient
* ``egrad(X, S, ar)``           Euclidean gradient of the computational extension F,
                                evaluated in the order of Lemma 3 with arithmetic ``ar``
* ``H(X, S)``                   damping matrix  H = -sym(X^T grad F(X; S))
* ``error(X)``                  ||sin Theta||_F to the reference subspace
* ``rgrad_norm2(X)``            ||grad f||_F^2 of the full objective at span(X)
* ``lam`` (Rayleigh problems)   eigenvalues of A, used by the a-priori bounds
"""
from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat


def orth(X):
    """Orthonormal basis of span(X) (polar factor)."""
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def sin_theta(U, X):
    Q = orth(X)
    return np.linalg.norm(Q - U @ (U.T @ Q))


# --------------------------------------------------------------------------
# Rayleigh-quotient problems  F(X) = -1/2 tr(X^T A X)
# --------------------------------------------------------------------------
class Rayleigh:
    """Dominant-subspace problem with a covariance known through samples."""

    kind = "rayleigh"

    def __init__(self, A=None, data=None, p=4, name="rayleigh", sampler=None, matvec=None,
                 Up=None, lam=None):
        self.name, self.p = name, p
        self.A, self.data, self._sampler, self._matvec = A, data, sampler, matvec
        if A is not None:
            w, V = np.linalg.eigh(A)
            idx = np.argsort(w)[::-1]
            self.lam, self.V = w[idx], V[:, idx]
            self.Up = self.V[:, :p]
            self.n = A.shape[0]
        else:
            self.Up, self.lam = Up, lam
            self.n = Up.shape[0]
        self.gap = self.lam[p - 1] - self.lam[p]

    # -- data ---------------------------------------------------------------
    def sample(self, rng, b):
        if b is None:
            return None
        if self._sampler is not None:
            return self._sampler(rng, b)
        if self.data is not None:                     # finite sum
            idx = rng.integers(0, self.data.shape[1], size=b)
            return self.data[:, idx]
        z = rng.standard_normal((self.n, b))          # a ~ N(0, A)
        return self.V @ (np.sqrt(np.maximum(self.lam, 0))[:, None] * (self.V.T @ z))

    def AX(self, X):
        return self._matvec(X) if self._matvec is not None else self.A @ X

    # -- gradient (Lemma 3 evaluation order) ----------------------------------
    def egrad(self, X, S, ar):
        if S is None:
            return -ar.fl(self.AX(X)) if ar.t is not None else -self.AX(X)
        b = S.shape[1]
        B = ar.matmul(S.T, X)
        return -ar.fl(ar.matmul(S, B) / b)

    def H(self, X, S):
        AX = self.AX(X) if S is None else S @ (S.T @ X) / S.shape[1]
        M = X.T @ AX
        return 0.5 * (M + M.T)

    # -- accuracy -----------------------------------------------------------
    def error(self, X):
        return sin_theta(self.Up, X)

    def rgrad_norm2(self, X):
        Q = orth(X)
        G = self.AX(Q)
        G = G - Q @ (Q.T @ G)
        return float(np.sum(G * G))

    def fval(self, X):
        Q = orth(X)
        return -0.5 * float(np.sum(Q * self.AX(Q)))

    def fstar(self):
        return -0.5 * float(np.sum(self.lam[: self.p]))


def synthetic(seed=0, n=60, p=4, tail=(0.2, 1.0)):
    """Problem (i): leading eigenvalues 4, 3.5, 3, 2.5, the rest U(tail)."""
    rng = np.random.default_rng(seed)
    lead = np.array([4.0, 3.5, 3.0, 2.5])[:p]
    rest = np.sort(rng.uniform(tail[0], tail[1], n - p))[::-1]
    lam = np.concatenate([lead, rest])
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (V * lam) @ V.T
    return Rayleigh(A=0.5 * (A + A.T), p=p, name=f"synthetic_n{n}_tail{tail[1]}")


def spiked(seed=0, n=10_000, p=10):
    """Problem (ii): A = 0.5 I + U diag(s) U^T, never formed explicitly."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    lead = np.linspace(4.0, 2.5, p)
    s = lead - 0.5
    lam = np.concatenate([lead, np.full(n - p, 0.5)])

    def sampler(r, b):
        return np.sqrt(0.5) * r.standard_normal((n, b)) + U @ (np.sqrt(s)[:, None] * r.standard_normal((p, b)))

    def matvec(X):
        return 0.5 * X + U @ (s[:, None] * (U.T @ X))

    return Rayleigh(p=p, name="spiked", sampler=sampler, matvec=matvec, Up=U, lam=lam)


def digits(p=5):
    """Problem (iv): UCI optical recognition of handwritten digits (8x8, n=64).

    The data are the copy bundled with scikit-learn (1797 samples), centred.
    """
    from sklearn.datasets import load_digits

    X = load_digits().data.T.astype(float)            # 64 x 1797
    X = X - X.mean(axis=1, keepdims=True)
    lam1 = np.linalg.eigvalsh(X @ X.T / X.shape[1]).max()
    X = X * np.sqrt(4.0 / lam1)                       # scale so that lambda_1 = 4 (gap ~ 0.23)
    A = X @ X.T / X.shape[1]
    return Rayleigh(A=A, data=X, p=p, name="digits")


def from_data(W, p, name):
    """PCA problem for a data matrix W (n x N), samples drawn from the columns."""
    W = W - W.mean(axis=1, keepdims=True)
    W = W * np.sqrt(4.0 / np.linalg.eigvalsh(W @ W.T / W.shape[1]).max())   # lambda_1 = 4
    return Rayleigh(A=W @ W.T / W.shape[1], data=W, p=p, name=name)


# --------------------------------------------------------------------------
# Hopkins 155 (real data, must be supplied by the user)
# --------------------------------------------------------------------------
HOPKINS_SEQUENCES = ("1R2RC", "1R2RCR")


def hopkins_problems(root=None, p=3, sequences=HOPKINS_SEQUENCES):
    """Per-motion PCA problems from Hopkins 155 sequences.

    Expects ``<root>/<seq>/<seq>_truth.mat`` (the original Hopkins 155 files,
    variables ``x`` (3 x P x F) and ``s`` (motion labels)).  Returns an empty
    list if the files are not present.
    """
    root = root or os.environ.get("PARSGD_HOPKINS") or os.path.join(os.path.dirname(__file__), "..", "data", "hopkins")
    probs = []
    for seq in sequences:
        f = os.path.join(root, seq, f"{seq}_truth.mat")
        if not os.path.exists(f):
            continue
        M = loadmat(f)
        x, s = M["x"], M["s"].ravel()
        P, F = x.shape[1], x.shape[2]
        W = np.concatenate([x[0], x[1]], axis=1).T     # 2F x P trajectory matrix
        for lab in np.unique(s):
            probs.append(from_data(W[:, s == lab], p, f"{seq}_motion{lab}"))
    return probs


# --------------------------------------------------------------------------
# Robust projection energy  F(X) = E phi(||X^T a||^2),  phi(s) = -eps^2 log(1 + s/eps^2)
# --------------------------------------------------------------------------
class Robust:
    kind = "robust"

    def __init__(self, seed=0, n=60, p=4, N=4000, eps=4.0):
        rng = np.random.default_rng(seed)
        base = synthetic(seed, n, p)
        self.n, self.p, self.eps2, self.name = n, p, eps ** 2, "robust"
        z = rng.standard_normal((n, N))
        self.data = base.V @ (np.sqrt(base.lam)[:, None] * (base.V.T @ z))
        self.lam = base.lam
        # heavy-tailed contamination: 5% of samples scaled up
        out = rng.random(N) < 0.05
        self.data[:, out] *= 6.0
        self.Up = self._reference(base.Up)

    def _w(self, B):
        return 1.0 / (1.0 + np.sum(B * B, axis=1) / self.eps2)

    def sample(self, rng, b):
        if b is None:
            return None
        return self.data[:, rng.integers(0, self.data.shape[1], size=b)]

    def egrad(self, X, S, ar):
        S = self.data if S is None else S
        B = ar.matmul(S.T, X)
        w = self._w(B)
        return -ar.fl(ar.matmul(S, w[:, None] * B) * (2.0 / S.shape[1]))

    def H(self, X, S):
        S = self.data if S is None else S
        B = S.T @ X
        M = (self._w(B)[:, None] * B).T @ B * (2.0 / S.shape[1])
        return 0.5 * (M + M.T)

    def _reference(self, X0):
        from .arith import Arith
        X, ar = X0.copy(), Arith()
        for _ in range(3000):
            G = self.egrad(X, None, ar)
            X = orth(X - 0.05 * (G - X @ (X.T @ G)))
        return X

    def error(self, X):
        return sin_theta(self.Up, X)

    def rgrad_norm2(self, X):
        from .arith import Arith
        Q = orth(X)
        G = self.egrad(Q, None, Arith())
        G = G - Q @ (Q.T @ G)
        return float(np.sum(G * G))


# --------------------------------------------------------------------------
# Non-damped extension  F2(X) = 1/4 ||X X^T - C||_F^2   (Proposition 1)
# --------------------------------------------------------------------------
class F2Extension:
    kind = "f2"

    def __init__(self, base, s=1.0):
        self.base, self.C, self.n, self.p = base, s * base.A, base.n, base.p
        self.name = f"F2_s{s}"
        self.Up, self.lam = base.Up, base.lam

    def sample(self, rng, b):
        return None

    def egrad(self, X, S, ar):
        return ar.fl((X @ X.T - self.C) @ X)

    def H(self, X, S):
        M = X.T @ self.C @ X - (X.T @ X) @ (X.T @ X)
        return 0.5 * (M + M.T)

    def error(self, X):
        return sin_theta(self.Up, X)

    def rgrad_norm2(self, X):
        return self.base.rgrad_norm2(X)
