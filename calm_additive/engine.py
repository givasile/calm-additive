"""Executable CALM — materialize a `CalmSequence` (or raw partitions) into a predictor.

`select_regions` returns a chain of CALM *snapshots* — analysis values with no
`predict`. This module closes the loop: it turns the partitions a chain (or
`find_regions`) produced into an executable model, in one of two fitting modes:

- ``curve_fitter="surrogate"`` — **free**: freeze the effect's cached curves
  (the same reads `explained_variance` scores), the per-region centering
  constants, and the least-squares offsets, and route new points through the
  leaf rules. No refit, no extra dependencies, no model calls.
- ``curve_fitter="ebm" | <MaskedFitter>`` — **refit**: build one
  masked column per (feature, region), then fit an interpretable additive
  model on the masked design (the paper's CALM). Implementations register
  themselves via `register_curve_fitter` (`calm_additive` registers the
  default ones); this module never imports their dependencies.

Three entry levels, lowest first:

```python
model = fit_chain(chain, X, y, curve_fitter="ebm")     # a chain you inspected
model = fit_chain(chain, curve_fitter="surrogate")     # no y needed
calm  = CALMRegressor(max_partition_depth=1, min_r2_gain=0.02)
calm.fit(X, y); calm.predict(X_new)                    # the sklearn twins
```

The twins take every setting as a keyword argument, in three tiers: the
hyperparameters to tune, the choices each step uses (a registered name or an
object, tuned through the matching `*_params` dict), and the advanced ones.
`selection` picks how the proposed partitions are kept (`"r2_gain"` =
`select_regions`, the method identity; `"all"` = `find_regions`, the pre-0.5
semantics — every found partition is applied, no cross-feature selection).
`selection_target` scores candidate partitions against the teacher
(`"teacher"`, default) or the labels (`"y"`).

This module imports only effector leaves + numpy. Heavy fitters live outside.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import time
from dataclasses import dataclass, fields, replace
from typing import Callable, Optional, Union

import numpy as np

from effector import ingestion
from effector.calm import CALM, CalmSequence

__all__ = [
    "MaskedFitter",
    "FitterContext",
    "Curve",
    "AdditiveTerms",
    "FeatureTerms",
    "ModelTerms",
    "register_curve_fitter",
    "register_teacher",
    "MaskedTransform",
    "SurrogateModel",
    "ExecutableCALM",
    "fit_chain",
    "fit_partitions",
    "CALMRegressor",
    "CALMClassifier",
]


# ---------------------------------------------------------------------------
# fitter protocol + registries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitterContext:
    """What a fitter factory gets to size itself: the masked design's shape."""

    task: str  # "regression" | "classification"
    dim: int  # number of masked columns
    feature_names: tuple
    feature_types: tuple  # per masked column, the source feature's type


@dataclass(frozen=True)
class Curve:
    """A one-dimensional function, the way a fitter stores it.

    | `kind` | `x` | `y` | value at `v` |
    |---|---|---|---|
    | `"step"` | `(B-1,)` ascending cut points | `(B,)` | `y[i]` on `[x[i-1], x[i])` |
    | `"linear"` | `(T,)` ascending knots | `(T,)` | linear between knots, flat outside |
    | `"levels"` | `(L,)` ascending levels | `(L,)` | `y` of the matching level, else `other` |

    `fn`, when given, is the exact evaluator and `x`/`y` only its drawable
    sampling (a curve read from an effector effect).
    """

    x: np.ndarray
    y: np.ndarray
    kind: str = "step"
    other: float = 0.0
    fn: Optional[Callable] = None

    def __call__(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if self.fn is not None:
            return np.asarray(self.fn(v), dtype=float)
        if self.kind == "step":
            return self.y[np.digitize(v, self.x)]
        if self.kind == "linear":
            return np.interp(v, self.x, self.y)
        out = np.full(v.shape, self.other, dtype=float)
        if len(self.x):
            idx = np.clip(np.searchsorted(self.x, v), 0, len(self.x) - 1)
            hit = self.x[idx] == v
            out[hit] = self.y[idx[hit]]
        return out

    def shifted(self, c: float) -> "Curve":
        """The same curve moved up by `c`."""
        fn = self.fn
        return replace(
            self,
            y=self.y + c,
            other=self.other + c,
            fn=None if fn is None else (lambda v: np.asarray(fn(v), dtype=float) + c),
        )

    @classmethod
    def constant(cls, c: float) -> "Curve":
        return cls(np.empty(0), np.array([float(c)]), "step")


@dataclass(frozen=True)
class AdditiveTerms:
    """What a `MaskedFitter` learned, column by column.

    The fitter's score is `intercept` plus, per masked column, the column's
    curve at the row's value when the column is active and `inactive[c]`
    when it is not. `link` names the scale that sum lives on: `"identity"`
    (the prediction itself) or `"logit"` (log-odds of the positive class).
    """

    intercept: float
    columns: list  # one Curve per masked column
    inactive: np.ndarray  # (D',) the score of an inactive column
    link: str = "identity"


@dataclass(frozen=True)
class FeatureTerms:
    """One feature of a fitted CALM: a curve per region.

    `curves[i]` is the feature's whole contribution inside `leaves[i]`
    (`leaves` is empty and `curves` has one entry when the feature is not
    split); `fallback` serves a row no region claims.
    """

    feature: int
    curves: list
    leaves: list
    fallback: Curve


@dataclass(frozen=True)
class ModelTerms:
    """A fitted CALM as functions: `score = intercept + Σ_j curve_j(x_j)`,
    the curve of feature `j` being the one of the region the row falls in."""

    intercept: float
    link: str
    features: list  # one FeatureTerms per feature, by index


class MaskedFitter:
    """The refit contract: an additive model over masked columns.

    `X` is the masked design (one column per (feature, region), inactive
    entries zeroed), `mask` the parallel boolean activity matrix. Implement
    `fit(X, y, mask)` and `predict(X, mask)`; `predict` must return `(N,)`
    scores (probabilities of the positive class for classification).

    Implement `curves()` as well — the fitted model as an `AdditiveTerms` —
    and the model can be printed and plotted (`print_summary`, `plot_curves`, …);
    without it the model still predicts.
    """

    def fit(self, X, y, mask):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X, mask):  # pragma: no cover - interface
        raise NotImplementedError

    def curves(self) -> AdditiveTerms:  # pragma: no cover - interface
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its curves; implement "
            f"`curves()` (see `MaskedFitter`) to print and plot the model"
        )


_CURVE_FITTERS: dict = {}
_TEACHERS: dict = {}


def register_curve_fitter(name: str, factory: Callable[..., MaskedFitter]):
    """Register a fitter factory under a name (`calm_additive` does this
    for `"ebm"`): `factory(ctx, **params)` → a `MaskedFitter`, where `params`
    is the user's `curve_fitter_params` (plus `random_state` if the factory
    accepts it). `"surrogate"` is built in and reserved."""
    if name == "surrogate":
        raise ValueError("'surrogate' is the built-in free mode; pick another name")
    _CURVE_FITTERS[name] = factory


def register_teacher(name: str, factory: Callable[..., object]):
    """Register a teacher factory: `factory(task, **params)` → object with
    `fit(X, y)` and `forward(X)` (and optionally `jac(X)` for derivative
    methods), where `params` is the user's `teacher_params` (plus
    `random_state` if the factory accepts it)."""
    _TEACHERS[name] = factory


def _build(factory, first, params: Optional[dict], random_state):
    """`factory(first, **params)`; the seed is added when the factory takes it
    and the user did not set one."""
    kwargs = dict(params or {})
    if random_state is not None and "random_state" not in kwargs:
        accepted = inspect.signature(factory).parameters.values()
        if any(
            p.name == "random_state" or p.kind is inspect.Parameter.VAR_KEYWORD
            for p in accepted
        ):
            kwargs["random_state"] = random_state
    return factory(first, **kwargs)


def _resolve_curve_fitter(spec, ctx: FitterContext, params=None, random_state=None):
    if isinstance(spec, str):
        if spec in _CURVE_FITTERS:
            return _build(_CURVE_FITTERS[spec], ctx, params, random_state)
        raise ValueError(
            f"unknown curve_fitter {spec!r}; registered: "
            f"{sorted(_CURVE_FITTERS) or 'none'}"
            " — see `calm_additive.register_curve_fitter`"
        )
    if params:
        raise ValueError(
            "curve_fitter_params tunes a fitter given by name; the fitter "
            "passed is an object and is used as it is"
        )
    if hasattr(spec, "fit") and hasattr(spec, "predict"):
        return spec
    raise ValueError(f"curve_fitter must be a name or a MaskedFitter; got {spec!r}")


# ---------------------------------------------------------------------------
# the masked design (the old data_transform, over Partition/Rule)
# ---------------------------------------------------------------------------


class MaskedTransform:
    """One column per (feature, region): the paper's masked design.

    A feature with a partition of L leaves contributes L columns (its values
    zeroed outside each leaf); an unpartitioned feature contributes one
    column. Routing is `leaf.rule.contains(X)` — the same predicate that
    defines the region — so the transform applies to unseen data.
    """

    def __init__(self, partitions: dict, dim: int, feature_names):
        self.partitions = {j: p for j, p in partitions.items() if len(p.leaves) > 1}
        self.dim = dim
        self.feature_names = list(feature_names)
        self.columns = []  # (feature, leaf_or_None, name)
        self.foi_2_cols: dict = {}
        for j in range(dim):
            cols = []
            part = self.partitions.get(j)
            if part is None:
                cols.append(len(self.columns))
                self.columns.append((j, None, self.feature_names[j]))
            else:
                for leaf in part.leaves:
                    cols.append(len(self.columns))
                    name = f"{self.feature_names[j]} | {leaf.rule.format(self.feature_names)}"
                    self.columns.append((j, leaf, name))
            self.foi_2_cols[j] = cols

    @property
    def new_names(self) -> list:
        return [name for _, _, name in self.columns]

    def transform(self, X: np.ndarray):
        """`X (N, D)` → masked design `(N, D')` + boolean activity mask."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        Xt = np.zeros((n, len(self.columns)))
        mask = np.zeros((n, len(self.columns)), dtype=bool)
        for c, (j, leaf, _) in enumerate(self.columns):
            if leaf is None:
                Xt[:, c] = X[:, j]
                mask[:, c] = True
            else:
                m = leaf.rule.contains(X)
                Xt[m, c] = X[m, j]
                mask[:, c] = m
        return Xt, mask

    @property
    def interactions_info(self) -> dict:
        """Counts for the paper's metrics: leaves per split feature and the
        conditioning sets `S_i` (n_conditional_interactions = Σ_i |S_i|)."""
        split_features = {
            j: sorted(
                {k for leaf in p.leaves for k in leaf.rule.conditions}
            )
            for j, p in self.partitions.items()
        }
        return {
            "n_features": self.dim,
            "n_columns": len(self.columns),
            "leaves": {j: len(p.leaves) for j, p in self.partitions.items()},
            "split_features": split_features,
            "n_conditional_interactions": sum(
                len(s) for s in split_features.values()
            ),
        }


# ---------------------------------------------------------------------------
# the free mode: the frozen surrogate
# ---------------------------------------------------------------------------


class SurrogateModel:
    """The additive surrogate, frozen into a standalone predictor.

    At fit: freeze each feature's curve parameters (per region where
    partitioned, plus a root fallback), the training-data centering
    constants, and the least-squares offsets — the exact quantities
    `explained_variance.surrogate_r2` scores. At predict: route new rows
    through the leaf rules and sum the frozen curve reads. Zero model calls.
    """

    def __init__(self):
        self._terms = {}  # feature -> list of (rule_or_None, params, center)
        self._root_terms = {}  # feature -> (params, center) fallback
        self._offset_parts = []  # list of list[Rule] (leaves[:-1] per partition)
        self._offset_features = []  # the split feature of each entry above
        self._beta = None
        self._intercept_only = None

    @staticmethod
    def _freeze_term(effect, feature: int, mask):
        rows = slice(None) if mask is None else mask
        xs = effect.data[rows, feature]
        try:
            params = effect._summary(feature, mask)
            mu = np.asarray(effect._eval_payload(feature, params, xs), dtype=float)
        except ValueError:
            return None
        return params, float(mu.mean())

    def fit(self, effect, partitions: dict, features: list, target: np.ndarray):
        """Freeze curves + offsets so that predictions reproduce the surrogate
        `g` that `surrogate_r2(effect, target, partitions, features)` scores."""
        target = np.asarray(target, dtype=float).reshape(-1)
        n = effect.data.shape[0]
        contrib = np.zeros(n)
        for j in features:
            part = partitions.get(j)
            root = self._freeze_term(effect, j, None)
            if root is not None:
                self._root_terms[j] = root
            if part is None:
                if root is not None:
                    self._terms[j] = [(None, *root)]
                    params, c = root
                    xs = effect.data[:, j]
                    contrib += (
                        np.asarray(effect._eval_payload(j, params, xs), dtype=float)
                        - c
                    )
            else:
                terms = []
                for leaf in part.leaves:
                    m = part.mask(leaf.idx)
                    t = self._freeze_term(effect, j, m)
                    if t is None:
                        continue
                    params, c = t
                    terms.append((leaf.rule, params, c))
                    xs = effect.data[m, j]
                    contrib[m] += (
                        np.asarray(effect._eval_payload(j, params, xs), dtype=float)
                        - c
                    )
                self._terms[j] = terms
        # the piecewise-constant term, mirroring explained_variance._fit_offsets
        residual = target - contrib
        if not partitions:
            self._intercept_only = float(residual.mean())
        else:
            cols = [np.ones(n)]
            self._offset_parts = []
            self._offset_features = list(partitions)
            for part in partitions.values():
                rules = [leaf.rule for leaf in part.leaves[:-1]]
                self._offset_parts.append(rules)
                for leaf in part.leaves[:-1]:
                    cols.append(part.mask(leaf.idx).astype(float))
            Z = np.column_stack(cols)
            self._beta, *_ = np.linalg.lstsq(Z, residual, rcond=None)
        self._effect = effect  # only for _eval_payload at predict time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        g = np.zeros(n)
        for j, terms in self._terms.items():
            done = np.zeros(n, dtype=bool)
            for rule, params, c in terms:
                if rule is None:
                    m = np.ones(n, dtype=bool)
                else:
                    m = rule.contains(X) & ~done
                if not m.any():
                    continue
                mu = np.asarray(
                    self._effect._eval_payload(j, params, X[m, j]), dtype=float
                )
                g[m] += mu - c
                done |= m
            # rows no leaf claims (e.g. unseen categorical level): root fallback
            if not done.all() and j in self._root_terms:
                params, c = self._root_terms[j]
                m = ~done
                mu = np.asarray(
                    self._effect._eval_payload(j, params, X[m, j]), dtype=float
                )
                g[m] += mu - c
        if self._intercept_only is not None:
            return g + self._intercept_only
        cols = [np.ones(n)]
        for rules in self._offset_parts:
            for rule in rules:
                cols.append(rule.contains(X).astype(float))
        return g + np.column_stack(cols) @ self._beta

    def terms(self, partitions: dict, nof_points: int = 200) -> ModelTerms:
        """The frozen model as curves: per feature and region, the centred
        curve read plus that region's offset. `nof_points` sizes the drawable
        sampling of a continuous curve; evaluation stays exact."""
        effect = self._effect
        # the offset of each (feature, leaf rule); the last leaf of a
        # partition is the reference and has none
        offsets, k = {}, 1
        for j, rules in zip(self._offset_features, self._offset_parts):
            for rule in rules:
                offsets[(j, rule)] = float(self._beta[k])
                k += 1
        intercept = (
            self._intercept_only
            if self._intercept_only is not None
            else float(self._beta[0])
        )

        def curve(j, term, offset=0.0):
            if term is None:  # the effect has no curve for this feature
                return Curve.constant(offset)
            params, c = term

            def fn(v, j=j, params=params, shift=offset - c):
                return np.asarray(effect._eval_payload(j, params, v), dtype=float) + shift

            if ingestion.is_categorical(effect.feature_types[j]):
                xs, kind = np.unique(effect.data[:, j]), "levels"
            else:
                lo, hi = effect.axis_limits[:, j]
                xs, kind = np.linspace(lo, hi, nof_points), "linear"
            return Curve(xs, fn(xs), kind, fn=fn)

        features = []
        for j in range(effect.dim):
            root = self._root_terms.get(j)
            part = partitions.get(j)
            if part is None:
                features.append(FeatureTerms(j, [curve(j, root)], [], curve(j, root)))
                continue
            frozen = {rule: (params, c) for rule, params, c in self._terms.get(j, [])}
            leaves = list(part.leaves)
            curves = [
                curve(j, frozen.get(leaf.rule, root), offsets.get((j, leaf.rule), 0.0))
                for leaf in leaves
            ]
            features.append(FeatureTerms(j, curves, leaves, curve(j, root)))
        return ModelTerms(float(intercept), "identity", features)


# ---------------------------------------------------------------------------
# the executable model + the two fit entry points
# ---------------------------------------------------------------------------


class ExecutableCALM:
    """A fitted, predict-capable CALM. Built by `fit_chain`/`fit_partitions`.

    Carries its provenance: `.chain` (when built from one), `.calm` (the
    snapshot), `.partitions`, `.transform` + `.curve_fitter` (refit mode) or
    `.surrogate` (free mode). `predict` returns labels for classification
    (`predict_score` gives the raw score / positive-class probability).
    """

    def __init__(
        self,
        task: str,
        *,
        partitions: dict,
        chain: Optional[CalmSequence] = None,
        calm: Optional[CALM] = None,
        surrogate: Optional[SurrogateModel] = None,
        transform: Optional[MaskedTransform] = None,
        curve_fitter: Optional[MaskedFitter] = None,
    ):
        self.task = task
        self.partitions = partitions
        self.chain = chain
        self.calm = calm
        self.surrogate = surrogate
        self.transform = transform
        self.curve_fitter = curve_fitter

    def predict_score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.surrogate is not None:
            return self.surrogate.predict(X)
        Xt, mask = self.transform.transform(X)
        return np.asarray(self.curve_fitter.predict(Xt, mask)).reshape(-1)

    def predict(self, X) -> np.ndarray:
        score = self.predict_score(X)
        if self.task == "classification":
            return (score > 0.5).astype(int)
        return score

    @property
    def interactions_info(self) -> dict:
        if self.transform is not None:
            return self.transform.interactions_info
        split_features = {
            j: sorted({k for leaf in p.leaves for k in leaf.rule.conditions})
            for j, p in self.partitions.items()
        }
        return {
            "n_features": None,
            "n_columns": None,
            "leaves": {j: len(p.leaves) for j, p in self.partitions.items()},
            "split_features": split_features,
            "n_conditional_interactions": sum(
                len(s) for s in split_features.values()
            ),
        }

    @property
    def n_conditional_interactions(self) -> int:
        """The paper's metric: Σ_i |S_i| over the split features — one per
        (feature, conditioning feature) pair, so `x1 | x2` and `x2 | x1`
        count as two."""
        return self.interactions_info["n_conditional_interactions"]

    def terms(self) -> ModelTerms:
        """The model as functions — one curve per (feature, region), each a
        feature's whole contribution inside its region. What `print_summary` and
        `plot_curves` read; the same for every curve fitter."""
        if self.surrogate is not None:
            return self.surrogate.terms(self.partitions)
        fitted = self.curve_fitter.curves()
        transform = self.transform
        features = []
        for j in range(transform.dim):
            cols = transform.foi_2_cols[j]
            total = float(sum(fitted.inactive[c] for c in cols))
            # inside a region the feature contributes its active column's
            # curve plus what its other, inactive columns add
            curves = [
                fitted.columns[c].shifted(total - float(fitted.inactive[c]))
                for c in cols
            ]
            leaves = [transform.columns[c][1] for c in cols]
            if leaves[0] is None:
                features.append(FeatureTerms(j, curves, [], curves[0]))
            else:
                features.append(FeatureTerms(j, curves, leaves, Curve.constant(total)))
        return ModelTerms(float(fitted.intercept), fitted.link, features)


def fit_partitions(
    effect,
    partitions: dict,
    *,
    task: str = "regression",
    curve_fitter="surrogate",
    curve_fitter_params: Optional[dict] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    target: str = "teacher",
    random_state: Optional[int] = None,
    chain: Optional[CalmSequence] = None,
    calm: Optional[CALM] = None,
) -> ExecutableCALM:
    """Materialize `{feature: Partition}` into an executable model.

    The lowest-level entry: both `fit_chain` and the sklearn twins land here.
    `partitions` must be bound to `effect`. For the surrogate mode the offset
    target is the teacher's predictions (``target="teacher"``) or `y`
    (``target="y"``, requires `y`). Refit modes require `y` (and use
    `X` — default `effect.data` — as the design source);
    `curve_fitter_params` and `random_state` reach a fitter given by name.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
    if target not in ("teacher", "y"):
        raise ValueError(f"target must be 'teacher' or 'y'; got {target!r}")
    partitions = {j: p for j, p in partitions.items() if len(p.leaves) > 1}

    if curve_fitter == "surrogate":
        if curve_fitter_params:
            raise ValueError("curve_fitter='surrogate' takes no curve_fitter_params")
        if target == "y":
            if y is None:
                raise ValueError("target='y' needs y")
            tvec = np.asarray(y, dtype=float).reshape(-1)
        else:
            if effect._y_pred is None:
                effect._y_pred = np.asarray(effect.model(effect.data))
            tvec = np.asarray(effect._y_pred, dtype=float).reshape(-1)
        features = [j for j in range(effect.dim)]
        surrogate = SurrogateModel().fit(effect, partitions, features, tvec)
        return ExecutableCALM(
            task, partitions=partitions, chain=chain, calm=calm, surrogate=surrogate
        )

    if y is None:
        raise ValueError(
            f"curve_fitter={curve_fitter!r} refits on labels — y is required"
        )
    X = effect.data if X is None else np.asarray(X, dtype=float)
    transform = MaskedTransform(partitions, effect.dim, effect.feature_names)
    Xt, mask = transform.transform(X)
    src_types = [
        effect.feature_types[j] for j, _, _ in transform.columns
    ]
    ctx = FitterContext(
        task=task,
        dim=Xt.shape[1],
        feature_names=tuple(transform.new_names),
        feature_types=tuple(src_types),
    )
    fitter_obj = _resolve_curve_fitter(
        curve_fitter, ctx, curve_fitter_params, random_state
    )
    fitter_obj.fit(Xt, np.asarray(y).reshape(-1), mask)
    return ExecutableCALM(
        task,
        partitions=partitions,
        chain=chain,
        calm=calm,
        transform=transform,
        curve_fitter=fitter_obj,
    )


def fit_chain(
    chain: CalmSequence,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    curve_fitter="surrogate",
    curve_fitter_params: Optional[dict] = None,
    snapshot: Union[str, int] = "final",
    task: str = "regression",
    target: str = "teacher",
    random_state: Optional[int] = None,
) -> ExecutableCALM:
    """Materialize a chain you inspected: `snapshot` (default the final CALM)
    picks which decision-sequence prefix becomes the model."""
    calm = chain.final if snapshot == "final" else chain[int(snapshot)]
    effect = calm._require_effect()
    return fit_partitions(
        effect,
        dict(calm.partitions),
        task=task,
        curve_fitter=curve_fitter,
        curve_fitter_params=curve_fitter_params,
        X=X,
        y=y,
        target=target,
        random_state=random_state,
        chain=chain,
        calm=calm,
    )


# ---------------------------------------------------------------------------
# the sklearn twins
# ---------------------------------------------------------------------------


def _as_forward(teacher, task: str) -> Callable:
    """A `(N,) = f(X)` view of whatever the teacher is."""
    if hasattr(teacher, "forward"):
        return teacher.forward
    if task == "classification" and hasattr(teacher, "predict_proba"):
        return lambda X: np.asarray(teacher.predict_proba(X))[:, 1]
    if task == "classification" and hasattr(teacher, "predict"):
        raise ValueError(
            f"a classification teacher must give the positive-class "
            f"probability (predict_proba or forward); {type(teacher).__name__} "
            f"has only predict, which returns labels. Pass a callable X -> "
            f"probabilities instead."
        )
    if hasattr(teacher, "predict"):
        return lambda X: np.asarray(teacher.predict(X)).reshape(-1)
    if callable(teacher):
        return teacher
    raise ValueError(
        f"teacher must expose forward/predict_proba/predict or be callable; "
        f"got {teacher!r}"
    )


class _Progress:
    """The stage bar `verbose >= 1` shows: one tick per stage of `fit`, the
    running stage in the description, the time each took in the postfix
    (`1/5: 1.1s 2/5: 3.2s …`)."""

    def __init__(self, nof_stages: int, enabled: bool):
        from tqdm.auto import tqdm

        self._bar = tqdm(
            total=nof_stages,
            disable=not enabled,
            bar_format="{desc:<28}{bar:20} {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
        )
        self._times = []

    @contextlib.contextmanager
    def stage(self, name: str):
        self._bar.set_description_str(f"CALM · {name}")
        t0 = time.perf_counter()
        yield
        self._times.append(
            f"{len(self._times) + 1}/{self._bar.total}: "
            f"{time.perf_counter() - t0:.1f}s"
        )
        self._bar.set_postfix_str(" ".join(self._times))
        self._bar.update(1)

    def close(self, done: bool):
        if done:
            self._bar.set_description_str("CALM · fitted")
        self._bar.close()


# the flat partitioner knobs: effector's name and default for each
_PARTITIONER_KNOBS = {
    "max_partition_depth": ("max_depth", 2),
    "min_heterogeneity_drop": ("min_heterogeneity_decrease_pcg", 0.05),
}
# what CALM sets itself when it builds the effect
_EFFECT_RESERVED = ("data", "model", "schema", "features")


class _CALMBase:
    """Teacher → effect → partition proposal → selection → curves, in one `fit`.

    ```python
    calm = CALMRegressor(max_partition_depth=1, min_r2_gain=0.02)
    calm.fit(X, y)          # X: numpy (+ schema=) or a pandas DataFrame
    calm.predict(X_new)
    calm.print_summary()    # the model card;  calm.plot_curves() draws the curves
    calm.chain_.show()      # the decision ledger (selection="r2_gain")
    ```

    `teacher` may be a *prefit* model (used as-is), a registered name
    (`"xgb"` built in — instantiated and fitted), or a bare callable (treated
    as the fitted forward). A model that is not a scikit-learn estimator
    counts as prefit only if it sets `is_fitted_ = True`.
    """

    _task = None

    def __init__(
        self,
        *,
        # hyperparameters
        max_conditional_interactions: Optional[int] = 10,
        max_partitions: Optional[int] = None,
        max_partition_depth: int = 2,
        min_heterogeneity_drop: float = 0.05,
        min_r2_gain: float = 0.01,
        # choices
        teacher="xgb",
        effect: Union[str, object] = "pdp",
        partitioner: Union[str, object] = "best",
        partition_features: Union[str, list] = "heterogeneous",
        conditioning_features: Union[str, list] = "all",
        selection: str = "r2_gain",
        curve_fitter="ebm",
        # advanced
        selection_target: str = "teacher",
        min_relative_r2_gain: Optional[float] = None,
        teacher_params: Optional[dict] = None,
        effect_params: Optional[dict] = None,
        partitioner_params: Optional[dict] = None,
        curve_fitter_params: Optional[dict] = None,
        schema=None,
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ):
        self.max_conditional_interactions = max_conditional_interactions
        self.max_partitions = max_partitions
        self.max_partition_depth = max_partition_depth
        self.min_heterogeneity_drop = min_heterogeneity_drop
        self.min_r2_gain = min_r2_gain
        self.teacher = teacher
        self.effect = effect
        self.partitioner = partitioner
        self.partition_features = partition_features
        self.conditioning_features = conditioning_features
        self.selection = selection
        self.curve_fitter = curve_fitter
        self.selection_target = selection_target
        self.min_relative_r2_gain = min_relative_r2_gain
        self.teacher_params = teacher_params
        self.effect_params = effect_params
        self.partitioner_params = partitioner_params
        self.curve_fitter_params = curve_fitter_params
        self.schema = schema
        self.random_state = random_state
        self.verbose = verbose

    @classmethod
    def paper(cls, **overrides):
        """The apples-to-apples preset: old-CALM behavior on the new engine.

        Every proposed partition is applied (no R² selection, no cap on the
        conditional interactions), `Best` partitioner, threshold proposer, numeric-only conditioning
        (categorical splits off), 20 candidate split points, depth 2. The
        drop threshold translates the paper's variance-scale ε=0.2 to the std
        scale: ``1 − √0.8 ≈ 0.106``. Keyword arguments override the preset.
        """
        preset = dict(
            max_conditional_interactions=None,
            selection="all",
            partitioner="best",
            max_partition_depth=2,
            min_heterogeneity_drop=1.0 - math.sqrt(0.8),
            conditioning_features="numeric",
            partitioner_params={
                "numerical_features_grid_size": 20,
                "continuous_proposer": "threshold",
                "search_partitions_when_categorical": False,
            },
        )
        return cls(**{**preset, **overrides})

    # -- fit --------------------------------------------------------------
    def fit(self, X, y):
        """Fit the teacher, propose and select the partitions, fit the curves.

        Args:
            X: `(N, D)` numpy array or pandas DataFrame. A DataFrame brings
                its column names and categorical columns along.
            y: `(N,)` target — values for regression, 0/1 for classification.

        Returns:
            `self`, fitted.
        """
        self._validate_params()
        X, schema = self._ingest(X)
        ingestion.check_finite(X, None if schema is None else schema.feature_names)
        self.n_features_in_ = X.shape[1]
        schema = self._name_target(schema, y)
        y = self._check_y(y, X.shape[0])
        # kept for the views: curves are drawn where the training rows are,
        # and the GAM reference of `print_summary(X, y)` is fitted on the same data
        self._X_fit, self._y_fit = X, y
        self._view_builder_cache = None
        self.__dict__.pop("gam_", None)

        selecting = self.selection == "r2_gain"
        progress = _Progress(5 if selecting else 4, enabled=self.verbose >= 1)
        done = False
        try:
            with progress.stage(f"teacher ({self._slot_name('teacher')})"):
                self.teacher_ = self._resolve_teacher(X, y)
            forward = _as_forward(self.teacher_, self._task)
            with progress.stage(f"effect ({self._slot_name('effect')})"):
                self.effect_ = self._make_effect(X, forward, schema)
            self.feature_names_in_ = np.array(
                self.effect_.feature_names, dtype=object
            )

            conditioning = self._resolve_conditioning(self.effect_)
            partitioner = self._make_partitioner()
            with progress.stage("propose partitions"):
                parts = self.effect_.find_regions(
                    features=self.partition_features,
                    finder=partitioner,
                    candidate_conditioning_features=conditioning,
                )
            if selecting:
                target = y if self.selection_target == "y" else None
                with progress.stage("select partitions"):
                    self.chain_ = self.effect_.select_regions(
                        parts,
                        min_r2_gain=self.min_r2_gain,
                        target=target,
                        relative_gain=self.min_relative_r2_gain,
                        max_conditional_interactions=self.max_conditional_interactions,
                        max_partitions=self.max_partitions,
                    )
                self.partitions_ = dict(self.chain_.final.partitions)
            else:
                self.partitions_ = {
                    self.effect_._resolve_feature(k): p
                    for k, p in parts.items()
                    if len(p.leaves) > 1
                }
                self.chain_ = None

            with progress.stage(f"curves ({self._slot_name('curve_fitter')})"):
                self.model_ = fit_partitions(
                    self.effect_,
                    self.partitions_,
                    task=self._task,
                    curve_fitter=self.curve_fitter,
                    curve_fitter_params=self.curve_fitter_params,
                    X=X,
                    y=y,
                    target=self.selection_target,
                    random_state=self.random_state,
                    chain=self.chain_,
                    calm=self.chain_.final if self.chain_ is not None else None,
                )
            done = True
        finally:
            progress.close(done)
        self.interactions_info_ = self.model_.interactions_info
        self.n_conditional_interactions_ = self.model_.n_conditional_interactions
        return self

    # -- predict ----------------------------------------------------------
    def predict(self, X):
        """Predict: values for regression, 0/1 labels for classification.

        Args:
            X: `(N, D)` array or DataFrame, with the columns `fit` saw.

        Returns:
            `(N,)` predictions.
        """
        return self.model_.predict(self._encode(X))

    def predict_score(self, X):
        """The raw score: the prediction for regression, the positive-class
        probability for classification.

        Args:
            X: `(N, D)` array or DataFrame, with the columns `fit` saw.

        Returns:
            `(N,)` scores.
        """
        return self.model_.predict_score(self._encode(X))

    # -- views ------------------------------------------------------------
    def print_summary(self, X=None, y=None, *, max_features: int = 20):
        """Print the model card: what this model is, on one screen.

        ```python
        calm.print_summary()                  # size, features by importance, partitions
        calm.print_summary(X_test, y_test)    # … and where it stands on that data
        ```

        The card lists the model's size (curves, partitions, conditional
        interactions), every feature with its importance and the features
        its curves depend on, and each partition as a tree. Importance is
        the mean |contribution| of the feature over the training rows, read
        from the fitted curves.

        With data, a *where it stands* block scores three models on it: a
        GAM (the same curve fitter with no partitions), this CALM, and the
        black-box teacher — and says how much of the GAM → black-box
        distance CALM covers. The GAM is fitted on the training data the
        first time it is needed and kept as `gam_`.

        Args:
            X: `(N, D)` array or DataFrame to score on, with the columns
                `fit` saw; held-out data is the honest choice. Default
                `None`: no scores.
            y: `(N,)` targets for `X`.
            max_features: rows of the feature table; the rest are summed up.
        """
        from calm_additive.render import text

        if (X is None) != (y is None):
            raise ValueError("pass both X and y to score the model, or neither")
        builder = self._view_builder()
        scores = None if X is None else self._scores(X, y)
        print(text.model_card(builder.model_view(scores), max_features), end="")

    def print_prediction(self, x, *, max_features: int = 10):
        """Print why the model predicts what it does for one row.

        ```python
        calm.print_prediction(X_test.iloc[0])
        ```

        A ledger with one line per feature: its value, the rule that
        selected its curve (for a feature with a partition), and what that
        curve adds to the baseline. The lines add up to the prediction
        exactly: `baseline + Σ contributions`. A classifier's lines are in
        log-odds, the scale it is additive on, and the last line maps their
        sum to the probability.

        Args:
            x: one row — a `(D,)` array, a pandas Series, or a one-row
                array / DataFrame, with the columns `fit` saw.
            max_features: lines shown, largest |contribution| first; the
                rest are summed in one line.
        """
        from calm_additive.render import text

        view = self._view_builder().prediction_view(self._encode_row(x))
        print(text.prediction(view, max_features), end="")

    def plot_curves(
        self,
        features=None,
        *,
        at=None,
        ncols: int = 3,
        share_y: bool = False,
        max_features: int = 12,
        show_plot: bool = True,
    ):
        """Draw the model: one panel per feature, one curve per region.

        ```python
        calm.plot_curves()                      # the most important features
        calm.plot_curves("hour")                # one feature, large
        calm.plot_curves(at=X_test.iloc[0])     # … with one row marked on its curves
        ```

        A curve is the feature's contribution to the prediction inside its
        region (the legend gives the region's rule and its number of
        training rows) and is drawn only where that region has data; the
        strip under a panel shows where the rows are. A prediction is the
        baseline plus one curve value per feature. Categorical features are
        drawn as one mark per level.

        Args:
            features: a name or index (one large panel), a list of them, or
                `None` (default): the `max_features` most important ones.
            at: one row (as in `print_prediction`). Its curve is
                highlighted in every panel, with a dot at the row's value.
            ncols: panels per row.
            share_y: if `True`, all panels share the y-axis, so a feature
                that matters little looks flat. Default `False`.
            max_features: how many features `features=None` draws.
            show_plot: if `True`, show the figure and return `None`; else
                return `(fig, axes)` — `(fig, ax)` for a single feature.
        """
        from calm_additive.render import mpl

        builder = self._view_builder()
        view = builder.model_view()
        single = isinstance(features, (str, int, np.integer))
        if features is None:
            chosen = list(view.order[:max_features])
        else:
            chosen = [
                self.effect_._resolve_feature(f)
                for f in ([features] if single else features)
            ]
        row = None if at is None else builder.prediction_view(self._encode_row(at))
        return mpl.plot_curves(
            view,
            chosen,
            at=row,
            ncols=ncols,
            share_y=share_y,
            single=single,
            show_plot=show_plot,
        )

    def plot_importance(self, *, max_features: int = 15, show_plot: bool = True):
        """Draw the features by importance, one bar each.

        Importance is the mean |contribution| of the feature over the
        training rows, in the units of the prediction (log-odds for a
        classifier). The bar of a feature with a partition is cut into one
        segment per region: how much of its importance each region carries.

        Args:
            max_features: bars drawn, most important first.
            show_plot: if `True`, show the figure and return `None`; else
                return `(fig, ax)`.
        """
        from calm_additive.render import mpl

        view = self._view_builder().model_view()
        return mpl.plot_importance(view, max_features=max_features, show_plot=show_plot)

    def plot_prediction(self, x, *, max_features: int = 10, show_plot: bool = True):
        """Draw one prediction as a waterfall: from the baseline, one bar per
        feature, to the prediction.

        The figure of `print_prediction`: the same numbers, largest
        |contribution| first, each bar labelled with the feature's value and
        the rule that selected its curve.

        Args:
            x: one row — a `(D,)` array, a pandas Series, or a one-row
                array / DataFrame, with the columns `fit` saw.
            max_features: bars drawn; the rest are summed in one bar.
            show_plot: if `True`, show the figure and return `None`; else
                return `(fig, ax)`.
        """
        from calm_additive.render import mpl

        view = self._view_builder().prediction_view(self._encode_row(x))
        return mpl.plot_prediction(view, max_features=max_features, show_plot=show_plot)

    # -- the data behind the views -----------------------------------------
    def curves(self) -> dict:
        """The fitted curves as arrays, grouped by feature — what `plot_curves` draws.

        Returns:
            `{feature_name: [{"region", "x", "y", "n"}, ...]}`, one entry per
                region of the feature (a single one, with region
                `"everywhere"`, when it has no partition). `x`, `y` trace
                the curve over the part of the axis where the region has
                training rows (a step curve repeats its cut points; a
                categorical feature has one point per level, `x` being the
                level's place on the axis and `"levels"` their names); `n` is the
                region's number of training rows.
        """
        out = {}
        for f in self._view_builder().model_view().features:
            entries = []
            for r in f.regions:
                entry = {"region": r.label, "x": r.x, "y": r.y, "n": r.n}
                if f.is_categorical:
                    at = np.searchsorted(f.level_positions, r.x)
                    entry["levels"] = [f.level_names[i] for i in at]
                entries.append(entry)
            out[f.name] = entries
        return out

    def contributions(self, X) -> np.ndarray:
        """What each feature adds to the baseline, row by row.

        ```python
        c = calm.contributions(X_test)
        c.sum(axis=1) + calm.baseline_      # the model's score for every row
        ```

        The score is the prediction for regression and the log-odds of the
        positive class for a classifier (`curve_fitter="ebm"`).

        Args:
            X: `(N, D)` array or DataFrame, with the columns `fit` saw.

        Returns:
            `(N, D)` array, one column per feature in `feature_names_in_`.
        """
        return self._view_builder().contributions(self._encode(X))

    def importances(self) -> np.ndarray:
        """Each feature's importance: its mean |contribution| over the
        training rows, read from the fitted curves.

        Returns:
            `(D,)` array, one value per feature in `feature_names_in_`.
        """
        return self._view_builder().importances()

    @property
    def baseline_(self) -> float:
        """The average score over the training rows; every prediction is this
        plus the row's contributions."""
        return self._view_builder().baseline

    def __getstate__(self):
        # the cached reading is rebuilt on demand and may hold closures
        state = self.__dict__.copy()
        if "_view_builder_cache" in state:
            state["_view_builder_cache"] = None
        return state

    def _view_builder(self):
        """The fitted model read for its reader — built once per fit."""
        if not hasattr(self, "model_"):
            raise RuntimeError(
                f"this {type(self).__name__} is not fitted yet; call fit(X, y) first"
            )
        if self._view_builder_cache is None:
            from calm_additive.views import ViewBuilder

            self._view_builder_cache = ViewBuilder(self)
        return self._view_builder_cache

    def _encode_row(self, x) -> np.ndarray:
        """One row, in any of its shapes → the `(D,)` encoded vector."""
        if hasattr(x, "to_frame") and not ingestion.is_dataframe(x):
            x = x.to_frame().T  # a pandas Series: one row of a DataFrame
        if not ingestion.is_dataframe(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)
        X = self._encode(x)
        if X.shape[0] != 1:
            raise ValueError(f"expected one row; got {X.shape[0]}")
        return X[0]

    def _scores(self, X, y):
        """GAM, CALM and the teacher, scored on `(X, y)`."""
        from calm_additive.views import score_table

        Xn = self._encode(X)
        y = self._check_y(y, Xn.shape[0])
        clf = self._task == "classification"

        def labels(score):
            score = np.asarray(score, dtype=float).reshape(-1)
            return (score > 0.5).astype(int) if clf else score

        predictions = {}
        if isinstance(self.curve_fitter, str):
            if not hasattr(self, "gam_"):
                print("fitting the GAM reference (first call only) …", flush=True)
                self.gam_ = fit_partitions(
                    self.effect_,
                    {},
                    task=self._task,
                    curve_fitter=self.curve_fitter,
                    curve_fitter_params=self.curve_fitter_params,
                    X=self._X_fit,
                    y=self._y_fit,
                    target=self.selection_target,
                    random_state=self.random_state,
                )
            note = f"{self.curve_fitter} curves, no partitions"
            predictions["gam"] = (note, self.gam_.predict(Xn))
        predictions["calm"] = ("", self.model_.predict(Xn))
        forward = _as_forward(self.teacher_, self._task)
        predictions["black box"] = (
            f"the teacher ({self._slot_name('teacher')})",
            labels(forward(Xn)),
        )
        return score_table(self._task, y, predictions)

    # -- plumbing ---------------------------------------------------------
    def _name_target(self, schema, y):
        """A pandas `y` names the target, unless the schema already does."""
        name = getattr(y, "name", None)
        if not isinstance(name, str):
            return schema
        if schema is None:
            return ingestion.Schema(target_name=name)
        return schema if schema.target_name is not None else replace(schema, target_name=name)
    def _slot_name(self, slot: str) -> str:
        """What a slot holds, for the progress bar: the registered name, or
        the object's class."""
        spec = getattr(self, slot)
        if isinstance(spec, str):
            return spec
        if isinstance(spec, type):
            return spec.__name__
        return type(spec).__name__

    def _validate_params(self):
        if isinstance(self.verbose, bool) or not isinstance(
            self.verbose, (int, np.integer)
        ) or self.verbose < 0:
            raise ValueError(f"verbose must be an integer >= 0; got {self.verbose!r}")
        if self.selection not in ("r2_gain", "all"):
            raise ValueError(
                f"selection must be 'r2_gain' or 'all'; got {self.selection!r}"
            )
        if self.selection_target not in ("teacher", "y"):
            raise ValueError(
                f"selection_target must be 'teacher' or 'y'; "
                f"got {self.selection_target!r}"
            )
        for cap in ("max_conditional_interactions", "max_partitions"):
            value = getattr(self, cap)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(
                    f"{cap} must be None or an integer >= 0; got {value!r}"
                )
            if self.selection == "all":
                raise ValueError(
                    f"selection='all' keeps every proposed partition, so "
                    f"{cap}={value!r} cannot be applied; set {cap}=None"
                )
        gain = self.min_relative_r2_gain
        if gain is not None and not (0.0 < gain <= 1.0):
            raise ValueError(
                f"min_relative_r2_gain must be in (0, 1] or None; got {gain!r}"
            )
        # a *_params dict tunes a component given by name (a class too, for
        # the teacher); an object is already configured
        by_name = {
            "teacher": (str, type),
            "effect": str,
            "partitioner": str,
            "curve_fitter": str,
        }
        for slot, kinds in by_name.items():
            if getattr(self, f"{slot}_params") and not isinstance(
                getattr(self, slot), kinds
            ):
                raise ValueError(
                    f"{slot}_params tunes a {slot} given by name; the {slot} "
                    f"passed is an object and is used as it is"
                )
        return self

    def _make_partitioner(self):
        """The configured partitioner (effector's finder), or the object
        passed."""
        spec = self.partitioner
        if not isinstance(spec, str):
            for flat, (_, default) in _PARTITIONER_KNOBS.items():
                if getattr(self, flat) != default:
                    raise ValueError(
                        f"{flat} applies to a partitioner given by name; "
                        f"set it on the partitioner object instead"
                    )
            return spec
        from effector import space_partitioning

        partitioners = {
            "best": space_partitioning.Best,
            "best_level_wise": space_partitioning.BestLevelWise,
        }
        if spec not in partitioners:
            raise ValueError(
                f"unknown partitioner {spec!r}; expected one of "
                f"{sorted(partitioners)} or a partitioner object"
            )
        knobs = dict(self.partitioner_params or {})
        for flat, (theirs, _) in _PARTITIONER_KNOBS.items():
            if theirs in knobs:
                raise ValueError(
                    f"set {flat}= on the estimator, not {theirs!r} in "
                    f"partitioner_params"
                )
            knobs[theirs] = getattr(self, flat)
        return partitioners[spec](**knobs)

    def _resolve_conditioning(self, effect) -> Union[str, list]:
        """`"numeric"` → the non-categorical feature indices, else pass through."""
        if self.conditioning_features != "numeric":
            return self.conditioning_features
        return [
            j
            for j in range(effect.dim)
            if not ingestion.is_categorical(effect.feature_types[j])
        ]

    def _ingest(self, X):
        """Fit-time ingestion → `(matrix, schema)`. A DataFrame brings its
        names, types and category levels; fields declared in `schema=`
        override them. The DataFrame's encoding is kept for `_encode`."""
        user = None if self.schema is None else ingestion._as_schema(self.schema)
        self.encoding_ = None
        if not ingestion.is_dataframe(X):
            return np.asarray(X, dtype=float), user
        kwargs = {}
        if user is not None and user.cat_limit is not None:
            kwargs["cat_limit"] = user.cat_limit
        data, schema, self.encoding_ = ingestion.from_dataframe(
            X, return_encoding=True, **kwargs
        )
        if user is not None:
            declared = {
                f.name: getattr(user, f.name)
                for f in fields(user)
                if getattr(user, f.name) is not None
            }
            schema = replace(schema, **declared)
        return data, schema

    def _encode(self, X):
        """Predict-time ingestion → the numeric matrix, encoded as at fit."""
        if ingestion.is_dataframe(X):
            X = self.encoding_.transform(X) if self.encoding_ is not None else X.to_numpy()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has shape {X.shape}; expected (N, {self.n_features_in_}) "
                f"as in fit"
            )
        ingestion.check_finite(X, self.feature_names_in_)
        return X

    def _check_y(self, y, n: int):
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n:
            raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("the target contains missing or infinite values")
        if self._task == "classification":
            levels = np.unique(y).tolist()
            if set(levels) != {0, 1}:
                raise ValueError(
                    f"the target must be binary, coded 0/1, with both classes "
                    f"present; got values {levels[:5]}"
                )
            y = y.astype(int)
        return y

    def _resolve_teacher(self, X, y):
        teacher = self.teacher
        created = False
        if isinstance(teacher, str):
            if teacher not in _TEACHERS:
                raise ValueError(
                    f"unknown teacher {teacher!r}; registered: "
                    f"{sorted(_TEACHERS) or 'none'} — see "
                    f"`calm_additive.register_teacher`"
                )
            teacher = _build(
                _TEACHERS[teacher], self._task, self.teacher_params, self.random_state
            )
            created = True
        elif isinstance(teacher, type):
            teacher = teacher(**(self.teacher_params or {}))
            created = True
        if hasattr(teacher, "fit") and (created or not self._looks_prefit(teacher)):
            teacher.fit(X, y)
        return teacher

    @staticmethod
    def _looks_prefit(teacher) -> bool:
        """Whether a user-passed *instance* can be trusted as already fitted.
        Registry names and classes never reach this — they are fresh and get
        fitted unconditionally. Pass a bare callable (e.g. `teacher.forward`)
        to use a prefit model with zero refit cost."""
        try:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted

            try:
                check_is_fitted(teacher)
                return True
            except NotFittedError:
                return False
            except Exception:
                pass  # not an sklearn estimator — fall through
        except ImportError:
            pass
        return bool(getattr(teacher, "is_fitted_", False))

    def _make_effect(self, X, forward, schema):
        import effector

        spec = self.effect
        if not isinstance(spec, str):
            return spec  # a prebuilt, fitted effect
        effects = {
            "pdp": effector.PDP,
            "ale": effector.ALE,
            "rhale": effector.RHALE,
            "shap": effector.ShapDP,
        }
        if spec not in effects:
            raise ValueError(
                f"unknown effect {spec!r}; expected one of {sorted(effects)} "
                f"or a fitted effect instance"
            )
        cls = effects[spec]
        params = dict(self.effect_params or {})
        jac = params.pop("model_jac", None) or getattr(self.teacher_, "jac", None)
        # effector splits an effect's settings between its constructor and
        # its fit; each key of effect_params goes where effector takes it
        init_names = set(inspect.signature(cls.__init__).parameters)
        fit_names = set(inspect.signature(cls.fit).parameters)
        init_kwargs = dict(
            schema=schema, nof_instances="all", random_state=self.random_state
        )
        fit_kwargs = {}
        for key, value in params.items():
            if key in _EFFECT_RESERVED:
                raise ValueError(f"effect_params cannot set {key!r}; CALM sets it")
            if key in init_names:
                init_kwargs[key] = value
            elif key in fit_names:
                fit_kwargs[key] = value
            else:
                raise ValueError(
                    f"effect={spec!r} has no setting {key!r} (effect_params)"
                )
        if spec == "rhale":
            if jac is None:
                raise ValueError(
                    "effect='rhale' needs a jacobian: a teacher exposing .jac, "
                    "or effect_params={'model_jac': f}"
                )
            eff = cls(X, forward, jac, **init_kwargs)
        else:
            eff = cls(X, forward, **init_kwargs)
        eff.fit(features="all", **fit_kwargs)
        return eff


class CALMRegressor(_CALMBase):
    """CALM for regression.

    ```python
    calm = CALMRegressor(max_partition_depth=2)
    calm.fit(X_train, y_train)              # numpy array or pandas DataFrame
    calm.predict(X_test)
    calm.print_summary(X_test, y_test)      # the model card, and where it stands
    calm.plot_curves()                      # one panel per feature, one curve per region
    calm.print_prediction(X_test.iloc[0])   # why this prediction
    calm.chain_.show()                      # why each partition was kept or rejected
    ```

    One `fit` runs the whole pipeline: fit the teacher, read it with a
    feature-effect method, propose and select the partitions, then fit one
    curve per (feature, region). Every argument is keyword-only and has a
    default.

    Args:
        max_conditional_interactions: (K in the paper) the most conditional
            interactions the model may have — one per (feature, conditioning
            feature) pair, so `x1 | x2` and `x2 | x1` are two. A feature's
            partition is kept whole or not at all, so selection stops before
            the partition that would exceed the limit. Default 10; `None`
            for no limit. Needs `selection="r2_gain"`.
        max_partitions: the most features that may get a partition;
            selection stops once it is reached. Default `None`, no limit.
            Needs `selection="r2_gain"`.
        max_partition_depth: (d_max in the paper) depth of each feature's
            partition tree, so a feature gets at most
            `2**max_partition_depth` curves and a region's rule has at most
            that many conditions. Default 2.
        min_heterogeneity_drop: (ε in the paper) inside one partition tree,
            the fraction by which a split must reduce the feature's
            heterogeneity to be made. Default 0.05. Higher means fewer
            splits.
        min_r2_gain: smallest R² gain (fraction of the target's variance,
            default 0.01 = 1 pt) a feature's partition must add, on top of
            the partitions already kept, to be kept. Higher means fewer
            partitions.
        teacher: the black-box model the partitions are read from. A
            registered name (`"xgb"`, default), an unfitted model or class
            (fitted here), a fitted model (used as is), or a plain callable
            `X -> scores`.
        effect: how the teacher is read to detect interactions — `"pdp"`
            (default), `"ale"`, `"shap"`, `"rhale"`, or a fitted effector
            effect.
        partitioner: what proposes a partition tree per feature — `"best"`
            (default), `"best_level_wise"`, or an effector finder object.
        partition_features: which features may get a partition —
            `"heterogeneous"` (default; heterogeneity at or above the
            median), `"all"`, or a list.
        conditioning_features: which features may appear in the region
            rules — `"all"` (default), `"numeric"`, or a list.
        selection: which proposed partitions are kept — `"r2_gain"`
            (default) keeps only those that make an additive surrogate
            explain the target noticeably better (see `min_r2_gain`), up to
            `max_conditional_interactions`; `"all"` keeps every one and
            takes no limit.
        curve_fitter: what is fitted inside the regions — `"ebm"` (default;
            an EBM without interactions, refit on the labels), `"surrogate"`
            (no refit: the teacher's regional curves are frozen into a
            model), or a `MaskedFitter`.

    Other Parameters:
        selection_target: what the R² of `selection` is measured against —
            the teacher's predictions (`"teacher"`, default) or the labels
            (`"y"`).
        min_relative_r2_gain: optional relaxed gate for nearly saturated
            models — the threshold becomes
            `min(min_r2_gain, min_relative_r2_gain * (1 - R²))`.
        teacher_params: keyword arguments for a teacher given by name or
            class, e.g. `{"n_estimators": 1000}` for `"xgb"`.
        effect_params: effector's settings for an effect given by name —
            constructor or `fit` arguments, e.g. `{"nof_instances": 10_000}`
            (CALM's default is all rows) or `{"model_jac": f}` for
            `"rhale"`.
        partitioner_params: effector's settings for a partitioner given by
            name, e.g. `{"min_samples_leaf": 50}`. The depth and the
            heterogeneity drop are set by the two arguments above.
        curve_fitter_params: keyword arguments for a fitter given by name,
            e.g. `{"max_bins": 64}` for `"ebm"`. The EBM's `interactions`
            is always 0.
        schema: feature names and types for a numpy `X`. A DataFrame brings
            its own.
        random_state: one seed, passed to every component built by name.
            Default 42.
        verbose: 0 (default) is silent; 1 shows a progress bar over the
            stages of `fit` (teacher, effect, propose, select, curves) with
            the time each took.

    Attributes:
        chain_: the decision ledger (`CalmSequence`); `None` when
            `selection="all"`.
        partitions_: `{feature index: Partition}` for the split features.
        model_: the fitted additive model (`ExecutableCALM`).
        teacher_: the fitted teacher.
        effect_: the effector effect the partitions were read from.
        n_conditional_interactions_: number of (feature, conditioning
            feature) pairs in the model; `x1 | x2` and `x2 | x1` are two.
        interactions_info_: regions per feature and their conditioning
            features.
        n_features_in_: number of features seen in `fit`.
        feature_names_in_: their names — the DataFrame's columns, the
            schema's `feature_names`, or `x_0, x_1, …`.
        baseline_: the average score over the training rows; a prediction
            is this plus the row's `contributions`.
        gam_: the GAM reference (the same curve fitter, no partitions) —
            set by the first `print_summary(X, y)`.
    """

    _task = "regression"


class CALMClassifier(_CALMBase):
    """CALM for binary classification.

    ```python
    calm = CALMClassifier(max_partition_depth=2)
    calm.fit(X_train, y_train)              # numpy array or pandas DataFrame
    calm.predict(X_test)
    calm.print_summary(X_test, y_test)      # the model card, and where it stands
    calm.plot_curves()                      # one panel per feature, one curve per region
    calm.print_prediction(X_test.iloc[0])   # why this prediction
    calm.chain_.show()                      # why each partition was kept or rejected
    ```

    One `fit` runs the whole pipeline: fit the teacher, read it with a
    feature-effect method, propose and select the partitions, then fit one
    curve per (feature, region). Every argument is keyword-only and has a
    default.

    The target must be 0/1. The teacher is read through its positive-class
    probability; `predict` thresholds the score at 0.5.

    Args:
        max_conditional_interactions: (K in the paper) the most conditional
            interactions the model may have — one per (feature, conditioning
            feature) pair, so `x1 | x2` and `x2 | x1` are two. A feature's
            partition is kept whole or not at all, so selection stops before
            the partition that would exceed the limit. Default 10; `None`
            for no limit. Needs `selection="r2_gain"`.
        max_partitions: the most features that may get a partition;
            selection stops once it is reached. Default `None`, no limit.
            Needs `selection="r2_gain"`.
        max_partition_depth: (d_max in the paper) depth of each feature's
            partition tree, so a feature gets at most
            `2**max_partition_depth` curves and a region's rule has at most
            that many conditions. Default 2.
        min_heterogeneity_drop: (ε in the paper) inside one partition tree,
            the fraction by which a split must reduce the feature's
            heterogeneity to be made. Default 0.05. Higher means fewer
            splits.
        min_r2_gain: smallest R² gain (fraction of the target's variance,
            default 0.01 = 1 pt) a feature's partition must add, on top of
            the partitions already kept, to be kept. Higher means fewer
            partitions.
        teacher: the black-box model the partitions are read from. A
            registered name (`"xgb"`, default), an unfitted model or class
            (fitted here), a fitted model (used as is), or a plain callable
            `X -> scores`.
        effect: how the teacher is read to detect interactions — `"pdp"`
            (default), `"ale"`, `"shap"`, `"rhale"`, or a fitted effector
            effect.
        partitioner: what proposes a partition tree per feature — `"best"`
            (default), `"best_level_wise"`, or an effector finder object.
        partition_features: which features may get a partition —
            `"heterogeneous"` (default; heterogeneity at or above the
            median), `"all"`, or a list.
        conditioning_features: which features may appear in the region
            rules — `"all"` (default), `"numeric"`, or a list.
        selection: which proposed partitions are kept — `"r2_gain"`
            (default) keeps only those that make an additive surrogate
            explain the target noticeably better (see `min_r2_gain`), up to
            `max_conditional_interactions`; `"all"` keeps every one and
            takes no limit.
        curve_fitter: what is fitted inside the regions — `"ebm"` (default;
            an EBM without interactions, refit on the labels), `"surrogate"`
            (no refit: the teacher's regional curves are frozen into a
            model), or a `MaskedFitter`.

    Other Parameters:
        selection_target: what the R² of `selection` is measured against —
            the teacher's predictions (`"teacher"`, default) or the labels
            (`"y"`).
        min_relative_r2_gain: optional relaxed gate for nearly saturated
            models — the threshold becomes
            `min(min_r2_gain, min_relative_r2_gain * (1 - R²))`.
        teacher_params: keyword arguments for a teacher given by name or
            class, e.g. `{"n_estimators": 1000}` for `"xgb"`.
        effect_params: effector's settings for an effect given by name —
            constructor or `fit` arguments, e.g. `{"nof_instances": 10_000}`
            (CALM's default is all rows) or `{"model_jac": f}` for
            `"rhale"`.
        partitioner_params: effector's settings for a partitioner given by
            name, e.g. `{"min_samples_leaf": 50}`. The depth and the
            heterogeneity drop are set by the two arguments above.
        curve_fitter_params: keyword arguments for a fitter given by name,
            e.g. `{"max_bins": 64}` for `"ebm"`. The EBM's `interactions`
            is always 0.
        schema: feature names and types for a numpy `X`. A DataFrame brings
            its own.
        random_state: one seed, passed to every component built by name.
            Default 42.
        verbose: 0 (default) is silent; 1 shows a progress bar over the
            stages of `fit` (teacher, effect, propose, select, curves) with
            the time each took.

    Attributes:
        chain_: the decision ledger (`CalmSequence`); `None` when
            `selection="all"`.
        partitions_: `{feature index: Partition}` for the split features.
        model_: the fitted additive model (`ExecutableCALM`).
        teacher_: the fitted teacher.
        effect_: the effector effect the partitions were read from.
        n_conditional_interactions_: number of (feature, conditioning
            feature) pairs in the model; `x1 | x2` and `x2 | x1` are two.
        interactions_info_: regions per feature and their conditioning
            features.
        n_features_in_: number of features seen in `fit`.
        feature_names_in_: their names — the DataFrame's columns, the
            schema's `feature_names`, or `x_0, x_1, …`.
        baseline_: the average score over the training rows; a prediction
            is this plus the row's `contributions`.
        gam_: the GAM reference (the same curve fitter, no partitions) —
            set by the first `print_summary(X, y)`.
    """

    _task = "classification"

    def predict_proba(self, X):
        """Class probabilities, in scikit-learn's two-column shape.

        Args:
            X: `(N, D)` array or DataFrame, with the columns `fit` saw.

        Returns:
            `(N, 2)` array: `[P(y=0), P(y=1)]`.
        """
        p = np.clip(self.predict_score(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])
