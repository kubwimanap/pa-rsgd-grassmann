"""PA-RSGD: Riemannian SGD on the Grassmann manifold under interacting
rounding, stochastic-gradient and retraction errors (reference implementation)."""
from .arith import Arith, round_bits
from .problems import synthetic, spiked, digits, Robust, F2Extension, hopkins_problems, orth
from .algorithms import (Retraction, rsgd, pa_rsgd, rsvrg, PAConfig, random_start,
                         near_solution, drift, direction)

__all__ = ["Arith", "round_bits", "synthetic", "spiked", "digits", "Robust", "F2Extension", "hopkins_problems",
           "orth", "Retraction", "rsgd", "pa_rsgd", "rsvrg", "PAConfig", "random_start", "near_solution", "drift",
           "direction"]
