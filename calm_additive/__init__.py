"""CALM: interpretable by design.

Accurate locally additive models with conditional feature effects.
"""

from calm_additive import fitters, teachers  # noqa: F401  (register "ebm", "xgb")
from calm_additive.engine import (
    AdditiveTerms,
    CALMClassifier,
    CALMRegressor,
    Curve,
    ExecutableCALM,
    MaskedFitter,
    fit_chain,
    fit_partitions,
    register_curve_fitter,
    register_teacher,
)

__version__ = "0.1.0"

__all__ = [
    "AdditiveTerms",
    "CALMClassifier",
    "CALMRegressor",
    "Curve",
    "ExecutableCALM",
    "MaskedFitter",
    "fit_chain",
    "fit_partitions",
    "register_curve_fitter",
    "register_teacher",
]
