"""Masked leaf fitters — the additive models CALM refits inside the regions.

A fitter sees the masked design: one column per (feature, region), with a
parallel activity mask. Registered name: ``"ebm"`` (an Explainable Boosting
Machine without interactions — the paper's default).
"""

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

from calm_additive.engine import (
    AdditiveTerms,
    Curve,
    MaskedFitter,
    register_curve_fitter,
)


def _ebm_kwargs(params):
    """The EBM's settings: `curve_fitter_params` over the defaults, with the
    interactions always off — that is what makes the model a CALM."""
    if "interactions" in params:
        raise ValueError(
            "curve_fitter_params cannot set 'interactions': CALM's EBM is "
            "fitted without interactions"
        )
    return {"random_state": 42, **params, "interactions": 0}


class _MaskedEBM:
    """EBM over the masked design: inactive entries become NaN, which the
    EBM handles natively as missing."""

    def _prepare_X(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data X must be numeric.")
        return X

    def _default_mask(self, X, mask):
        if mask is None:
            return np.ones_like(X)
        return self._prepare_X(np.asarray(mask, dtype=float))

    def _nan_masked(self, X, mask):
        X_copy = X.copy()
        X_copy[mask == 0] = np.nan
        return X_copy

    def fit(self, X, y, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        self.model.fit(self._nan_masked(X, mask), y)

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        return self.model.predict(self._nan_masked(X, mask))

    def curves(self, link="identity"):
        """The fitted EBM as `AdditiveTerms`: per masked column its term (a
        step function over the EBM's bins, or one score per level for a
        nominal column), and the score of the EBM's "missing" bin — what
        the column adds when it is inactive."""
        ebm = self.model
        columns = [None] * len(ebm.bins_)
        inactive = np.zeros(len(ebm.bins_))
        for features, scores in zip(ebm.term_features_, ebm.term_scores_):
            (c,) = features  # interactions=0: one column per term
            scores = np.asarray(scores, dtype=float)
            bins = ebm.bins_[c][0]
            inactive[c] = scores[0]
            if isinstance(bins, dict):  # nominal: {level: bin index}
                levels = np.array([float(k) for k in bins])
                order = np.argsort(levels)
                index = np.array(list(bins.values()))[order]
                columns[c] = Curve(
                    levels[order], scores[index], "levels", other=float(scores[-1])
                )
            else:  # continuous: the cut points; scores[0] missing, [-1] unseen
                columns[c] = Curve(np.asarray(bins, dtype=float), scores[1:-1], "step")
        # a column the EBM kept no term for contributes nothing
        columns = [Curve.constant(0.0) if col is None else col for col in columns]
        intercept = float(np.ravel(ebm.intercept_)[0])
        return AdditiveTerms(intercept, columns, inactive, link)


class EBMRegressor(_MaskedEBM):
    def __init__(self, **kwargs):
        self.model = ExplainableBoostingRegressor(**_ebm_kwargs(kwargs))


class EBMClassifier(_MaskedEBM):
    def __init__(self, **kwargs):
        self.model = ExplainableBoostingClassifier(**_ebm_kwargs(kwargs))

    def predict_proba(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        return self.model.predict_proba(self._nan_masked(X, mask))[:, 1]

    def predict(self, X, mask=None):
        return (self.predict_proba(X, mask) >= 0.5).astype(int)


class ScoreAdapter(MaskedFitter):
    """Bridge to the engine: predict returns raw scores (probability of the
    positive class for classification), the executable layer thresholds."""

    def __init__(self, impl, classification: bool):
        self.impl = impl
        self.classification = classification

    def fit(self, X, y, mask):
        self.impl.fit(X, y, mask=mask)
        return self

    def predict(self, X, mask):
        if self.classification:
            return np.asarray(self.impl.predict_proba(X, mask)).reshape(-1)
        return np.asarray(self.impl.predict(X, mask)).reshape(-1)

    def curves(self):
        # a classifier's terms add up on the log-odds scale
        return self.impl.curves("logit" if self.classification else "identity")


def _ebm(ctx, **params):
    clf = ctx.task == "classification"
    impl = EBMClassifier(**params) if clf else EBMRegressor(**params)
    return ScoreAdapter(impl, clf)


register_curve_fitter("ebm", _ebm)
