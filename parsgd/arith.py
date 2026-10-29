"""Simulated reduced-precision arithmetic.

Every arithmetic result can be passed through a rounding map ``fl`` that keeps
``t`` significand bits, either by round-to-nearest (RN, unit roundoff u = 2^-t)
or by stochastic rounding (SR, unbiased, u = one unit in the last place = 2^(1-t)).
A wide exponent range is assumed (all quantities in the experiments are O(1)).

The class :class:`Arith` bundles the rounding mode with the elementary
operations used by the RSGD iteration so that the evaluation order of
Lemma 3 of the paper is reproduced exactly.
"""
from __future__ import annotations

import numpy as np


def round_bits(x, t, mode="RN", rng=None):
    """Round ``x`` to ``t`` significand bits (RN or SR)."""
    x = np.asarray(x, dtype=np.float64)
    if t is None:
        return x
    m, e = np.frexp(x)                     # x = m * 2**e, 0.5 <= |m| < 1
    s = np.ldexp(m, t)                     # significand scaled to integer range
    if mode == "RN":
        r = np.rint(s)
    elif mode == "SR":
        if rng is None:
            raise ValueError("stochastic rounding needs an rng")
        fl = np.floor(s)
        r = fl + (rng.random(s.shape) < (s - fl))
    else:
        raise ValueError(mode)
    return np.ldexp(r, e - t)


def to_binary16(x):
    """Round to IEEE binary16 (finite exponent range, subnormals) by RN."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


class Arith:
    """Arithmetic context.

    Parameters
    ----------
    t : int or None
        Significand bits; ``None`` means exact (float64) arithmetic.
    mode : {"RN", "SR"}
    fmt : {None, "binary16", "bfloat16"}
        Use a concrete floating-point format instead of ``t``.
    accumulate : {"wide", "low"}
        "wide": products/sums are formed exactly and the result is rounded;
        "low": every partial sum of a matrix product is rounded (sequential
        low-precision accumulation).
    """

    def __init__(self, t=None, mode="RN", rng=None, fmt=None, accumulate="wide"):
        self.mode, self.rng, self.fmt, self.accumulate = mode, rng, fmt, accumulate
        if fmt == "binary16":
            t = 11
        elif fmt == "bfloat16":
            t = 8
        self.t = t

    # unit roundoff used by the theory and by PA-RSGD
    @property
    def u(self):
        if self.t is None:
            return 0.0
        return 2.0 ** (-self.t) if self.mode == "RN" else 2.0 ** (1 - self.t)

    @property
    def exact(self):
        return self.t is None

    def fl(self, x):
        if self.t is None:
            return np.asarray(x, dtype=np.float64)
        if self.fmt == "binary16" and self.mode == "RN":
            return to_binary16(x)
        return round_bits(x, self.t, self.mode, self.rng)

    def matmul(self, A, B):
        if self.t is None:
            return A @ B
        if self.accumulate == "wide":
            return self.fl(A @ B)
        # sequential low-precision accumulation over the inner dimension
        C = np.zeros((A.shape[0], B.shape[1]))
        for k in range(A.shape[1]):
            C = self.fl(C + self.fl(np.outer(A[:, k], B[k, :])))
        return C

    def add(self, A, B):
        return self.fl(A + B)

    def scale(self, a, A):
        return self.fl(a * A)


class NativeArith(Arith):
    """Arithmetic through XLA's native reduced-precision kernels (JAX, CPU).

    Every stored quantity is cast to the format and matrix products are
    computed by ``jax.numpy.matmul`` on operands of that format.  Requires
    ``jax``; used only for the native-kernel rows of Table S5.
    """

    def __init__(self, fmt="bfloat16", rng=None):
        super().__init__(fmt=fmt, mode="RN", rng=rng)
        import jax.numpy as jnp

        self._jnp = jnp
        self._dt = jnp.bfloat16 if fmt == "bfloat16" else jnp.float16

    def fl(self, x):
        jnp = self._jnp
        return np.asarray(jnp.asarray(np.asarray(x, np.float32)).astype(self._dt), dtype=np.float64)

    def matmul(self, A, B):
        jnp = self._jnp
        a = jnp.asarray(np.asarray(A, np.float32)).astype(self._dt)
        b = jnp.asarray(np.asarray(B, np.float32)).astype(self._dt)
        return np.asarray(jnp.matmul(a, b), dtype=np.float64)
