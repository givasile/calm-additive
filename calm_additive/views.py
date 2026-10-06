"""What a fitted CALM says, as plain values.

A fitted CALM is `score = baseline + Σ_j curve_j(x_j)`, the curve of feature
`j` being the one of the region the row falls in. This module reads that out
of the fitted model once — the curves, each row's contributions, the
importances — and hands it on as values made of numbers and strings only:

```
fitted estimator ──► ViewBuilder ──► ModelView / PredictionView ──► render/
```

Nothing here draws or prints, and nothing here knows which curve fitter
produced the curves (it reads `model_.terms()`), so the look can change
without touching the numbers and a new fitter gets every view for free.
Region rules, level names and trees are formatted by effector's `Partition`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from effector import ingestion

__all__ = [
    "ViewBuilder",
    "ModelView",
    "FeatureView",
    "RegionView",
    "PredictionView",
    "ContributionView",
    "Scores",
    "score_table",
]

# density bins of a continuous feature; an integer-valued one with at most
# this many distinct values gets one bin per integer instead
_NOF_BINS = 40


# ---------------------------------------------------------------------------
# the values handed to the renderers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionView:
    """One curve of a feature: its contribution inside one region.

    `x`, `y` are drawable as they are — a polyline over the part of the axis
    where the region has training rows (a step curve repeats its cut points),
    or one point per level for a categorical feature. `density` counts the
    region's training rows per bin of the feature's `bins` (per level for a
    categorical feature).
    """

    conditions: tuple  # the rule, one formatted condition per feature
    n: int  # training rows in the region
    x: np.ndarray
    y: np.ndarray
    density: np.ndarray
    share: float  # the region's part of the feature's importance

    @property
    def label(self) -> str:
        return ", ".join(self.conditions) if self.conditions else "everywhere"


@dataclass(frozen=True)
class FeatureView:
    """One feature of the model: a curve per region, and how much it matters."""

    index: int
    name: str
    kind: str  # "continuous" | "ordinal" | "nominal"
    importance: float  # mean |contribution| over the training rows
    regions: tuple  # RegionView, in the partition's leaf order
    conditioned_on: tuple  # names of the features in its rules
    tree: tuple  # the partition tree as lines; () when not split
    bins: Optional[np.ndarray] = None  # continuous: density bin edges
    levels: Optional[np.ndarray] = None  # categorical: the level values
    level_names: Optional[tuple] = None  # categorical: one label per level
    # categorical: where each level sits on the axis — its rank, or its value
    # for an ordinal feature whose levels are plain numbers — and the tick
    # labels (`None`: the positions themselves are the labels)
    level_positions: Optional[np.ndarray] = None
    level_labels: Optional[tuple] = None

    @property
    def is_categorical(self) -> bool:
        return self.levels is not None

    @property
    def is_split(self) -> bool:
        return len(self.regions) > 1


@dataclass(frozen=True)
class Scores:
    """Where the model stands between a GAM and the black box, on some data.

    `rows` holds `(name, note, values)` per model, in the order GAM, CALM,
    black box (a model that could not be scored is absent); `values` follow
    `metrics`. `gap` is the part of the GAM → black-box distance that CALM
    covers on `gap_metric`, `None` when the black box is not ahead of the GAM.
    """

    n_rows: int
    metrics: tuple
    rows: tuple
    gap: Optional[float]
    gap_metric: str


@dataclass(frozen=True)
class ModelView:
    """The whole model: what `print_summary`, `plot_curves` and `plot_importance` show."""

    task: str  # "regression" | "classification"
    target_name: str
    unit: str  # what a contribution is measured in
    n_rows: int
    baseline: float  # the average score; contributions add to it
    features: tuple  # FeatureView, by feature index
    order: tuple  # feature indices, most important first
    n_curves: int
    n_partitions: int
    n_conditional_interactions: int
    scores: Optional[Scores] = None


@dataclass(frozen=True)
class ContributionView:
    """One feature's part of one prediction."""

    feature: int
    name: str
    value: float
    value_text: str  # the value as shown: a level name, an integer, …
    conditions: tuple  # the rule that selected the curve; () when not split
    region: int  # index into the feature's regions; -1 if no region claims the row
    contribution: float


@dataclass(frozen=True)
class PredictionView:
    """One prediction, taken apart: `score = baseline + Σ contributions`.

    `score` lives on the scale the model is additive on (`unit`). For a
    classifier fitted on log-odds, `prediction` is the probability the score
    maps to and `label` the predicted class; otherwise `prediction == score`.
    """

    task: str
    target_name: str
    unit: str
    baseline: float
    score: float
    prediction: float
    label: Optional[int]
    contributions: tuple  # ContributionView, largest |contribution| first


# ---------------------------------------------------------------------------
# reading the fitted model
# ---------------------------------------------------------------------------


def _polyline(curve, lo: float, hi: float):
    """`curve` between `lo` and `hi` as drawable points."""
    if hi <= lo:
        xs = np.array([lo])
        return xs, curve(xs)
    if curve.fn is None and curve.kind == "step":
        cuts = curve.x[(curve.x > lo) & (curve.x < hi)]
        edges = np.concatenate([[lo], cuts, [hi]])
        values = curve((edges[:-1] + edges[1:]) / 2)
        return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)
    inner = curve.x[(curve.x > lo) & (curve.x < hi)]
    xs = np.concatenate([[lo], inner, [hi]])
    return xs, curve(xs)


def _number(v: float) -> str:
    """A feature value as shown in a table: integers bare, else 4 digits."""
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{int(round(v)):,}" if abs(v) >= 10_000 else f"{int(round(v))}"
    return f"{v:.4g}"


class ViewBuilder:
    """Reads a fitted `CALMRegressor` / `CALMClassifier` for its reader.

    Built once per fit (the estimator caches it). Curves are re-centred so
    that every feature's contribution averages zero over the training rows;
    the constant this moves goes to `baseline`, which is then the average
    score. For the EBM fitter the shift is numerically nil — its terms are
    centred already.

    `contributions`, `importances` and `baseline` are in the model's own
    units and add up to its score. The views (`model_view`,
    `prediction_view`) are in display units: a `scale_x_list` / `scale_y`
    declared in the schema is applied there, as effector does on its plots.
    """

    def __init__(self, calm):
        model = calm.model_
        effect = calm.effect_
        terms = model.terms()
        X = calm._X_fit
        n, dim = X.shape

        self.task = calm._task
        self.link = terms.link
        self.partitions = model.partitions
        self.meta = effect.feature_metadata
        self.names = list(effect.feature_names)
        self.n_conditional_interactions = model.n_conditional_interactions
        self._X = X
        self._scale_x = effect.scale_x_list or [None] * dim
        # a y scale only makes sense where the score is the target itself
        scale_y = effect.scale_y if self.task == "regression" else None
        self._y_std = float(scale_y["std"]) if scale_y else 1.0
        self._y_mean = float(scale_y["mean"]) if scale_y else 0.0

        self.baseline = float(terms.intercept)
        self._curves, self._rules, self._fallbacks = [], [], []
        self._leaf_idx = {}  # feature -> the partition's region index per curve
        self._train_contrib = np.empty((n, dim))
        self._train_region = np.zeros((n, dim), dtype=int)
        for ft in terms.features:
            j = ft.feature
            rules = [leaf.rule for leaf in ft.leaves]
            c, region = self._route(ft.curves, rules, ft.fallback, X, j)
            shift = float(c.mean())
            self._curves.append([curve.shifted(-shift) for curve in ft.curves])
            self._fallbacks.append(ft.fallback.shifted(-shift))
            self._rules.append(rules)
            self._leaf_idx[j] = [leaf.idx for leaf in ft.leaves]
            self.baseline += shift
            self._train_contrib[:, j] = c - shift
            self._train_region[:, j] = region

    # -- numbers, in the model's units --------------------------------------
    @staticmethod
    def _route(curves, rules, fallback, X, j):
        """Feature `j`'s contribution on the rows of `X`, and each row's region."""
        x = X[:, j]
        if not rules:
            return np.asarray(curves[0](x), dtype=float), np.zeros(len(x), dtype=int)
        out = np.asarray(fallback(x), dtype=float).copy()
        region = np.full(len(x), -1, dtype=int)
        for i, (curve, rule) in enumerate(zip(curves, rules)):
            m = rule.contains(X) & (region < 0)
            if m.any():
                out[m] = curve(x[m])
                region[m] = i
        return out, region

    def contributions(self, X, return_regions: bool = False):
        """`(N, D)`: what each feature adds to the baseline on each row."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape)
        regions = np.zeros(X.shape, dtype=int)
        for j in range(X.shape[1]):
            out[:, j], regions[:, j] = self._route(
                self._curves[j], self._rules[j], self._fallbacks[j], X, j
            )
        return (out, regions) if return_regions else out

    def importances(self) -> np.ndarray:
        """`(D,)`: mean |contribution| of each feature over the training rows."""
        return np.abs(self._train_contrib).mean(axis=0)

    # -- display units -------------------------------------------------------
    @property
    def unit(self) -> str:
        if self.link == "logit":
            return "log-odds"
        return "probability" if self.task == "classification" else self.meta.target_name

    def _show_x(self, j, x):
        s = self._scale_x[j]
        return x if s is None else s["std"] * np.asarray(x, dtype=float) + s["mean"]

    def _level_text(self, j, value: float) -> str:
        """A categorical level as shown: its declared name, else the number."""
        names = (self.meta.category_names or {}).get(j)
        if names and float(value) in names:
            return str(names[float(value)])
        return _number(float(self._show_x(j, value)))

    def _feature_view(self, j) -> FeatureView:
        x = self._X[:, j]
        shown = self._show_x(j, x)
        n = len(x)
        kind = self.meta.feature_types[j]
        categorical = ingestion.is_categorical(kind)
        region_of = self._train_region[:, j]
        contrib = self._train_contrib[:, j]
        part = self.partitions.get(j)

        bins = levels = level_names = positions = labels = None
        half = 0.0  # half a unit of the axis, in the model's units
        if categorical:
            levels = np.unique(x)
            level_names = tuple(self._level_text(j, v) for v in levels)
            named = bool((self.meta.category_names or {}).get(j))
            if kind == ingestion.NOMINAL or named:
                positions, labels = np.arange(len(levels), dtype=float), level_names
            else:  # ordered numbers: the axis is the number line
                positions = np.asarray(self._show_x(j, levels), dtype=float)
        else:
            lo, hi = float(shown.min()), float(shown.max())
            integers = np.all(np.abs(shown - np.round(shown)) < 1e-6)
            if integers and hi - lo < 60:
                # one bin per integer, and curves drawn half a unit past the
                # region's first and last value: each step sits on its integer
                bins = np.arange(round(lo) - 0.5, round(hi) + 1.0)
                scale = self._scale_x[j]
                half = 0.5 / (abs(scale["std"]) if scale else 1.0)
            else:
                bins = np.linspace(lo, hi, _NOF_BINS + 1)

        regions = []
        for i, curve in enumerate(self._curves[j]):
            mask = region_of == i
            xr = x[mask]
            if not mask.any():  # no training row falls in this region
                xs = ys = np.empty(0)
                density = np.zeros(len(levels) if categorical else len(bins) - 1)
            elif categorical:
                xs = np.unique(xr)
                ys = curve(xs)
                density = np.array([(xr == v).sum() for v in levels])
                xs = positions[np.searchsorted(levels, xs)]
            else:
                xs, ys = _polyline(
                    curve, float(xr.min()) - half, float(xr.max()) + half
                )
                xs = self._show_x(j, xs)
                density = np.histogram(shown[mask], bins=bins)[0]
            conditions = ()
            if part is not None:
                conditions = tuple(part.conditions(self._leaf_idx[j][i]))
            regions.append(
                RegionView(
                    conditions=conditions,
                    n=int(mask.sum()),
                    x=np.asarray(xs, dtype=float),
                    y=np.asarray(ys, dtype=float) * self._y_std,
                    density=density,
                    share=float(np.abs(contrib[mask]).sum() / n) * self._y_std,
                )
            )

        conditioned_on, tree = (), ()
        if part is not None:
            seen = {}
            for rule in self._rules[j]:
                for f in rule.features:
                    seen.setdefault(f, self.names[f])
            conditioned_on = tuple(seen.values())
            tree = tuple(part.tree_lines())
        return FeatureView(
            index=j,
            name=self.names[j],
            kind=kind,
            importance=float(np.abs(contrib).mean()) * self._y_std,
            regions=tuple(regions),
            conditioned_on=conditioned_on,
            tree=tree,
            bins=bins,
            levels=levels,
            level_names=level_names,
            level_positions=positions,
            level_labels=labels,
        )

    def model_view(self, scores: Optional[Scores] = None) -> ModelView:
        """The model as the views show it."""
        features = tuple(self._feature_view(j) for j in range(len(self.names)))
        order = sorted(
            range(len(features)), key=lambda j: (-features[j].importance, j)
        )
        return ModelView(
            task=self.task,
            target_name=self.meta.target_name,
            unit=self.unit,
            n_rows=self._X.shape[0],
            baseline=self.baseline * self._y_std + self._y_mean,
            features=features,
            order=tuple(order),
            n_curves=sum(len(f.regions) for f in features),
            n_partitions=sum(f.is_split for f in features),
            n_conditional_interactions=self.n_conditional_interactions,
            scores=scores,
        )

    def prediction_view(self, x) -> PredictionView:
        """One row `(D,)` of the encoded matrix, taken apart."""
        X = np.asarray(x, dtype=float).reshape(1, -1)
        contrib, regions = self.contributions(X, return_regions=True)
        contrib, regions = contrib[0], regions[0]
        score = self.baseline + float(contrib.sum())

        rows = []
        for j, name in enumerate(self.names):
            part = self.partitions.get(j)
            conditions = ()
            if part is not None:
                if regions[j] >= 0:
                    leaf = self._leaf_idx[j][regions[j]]
                    conditions = tuple(part.conditions(leaf))
                else:
                    conditions = ("outside every region",)
            value = float(X[0, j])
            if ingestion.is_categorical(self.meta.feature_types[j]):
                text = self._level_text(j, value)
            else:
                text = _number(float(self._show_x(j, value)))
            rows.append(
                ContributionView(
                    feature=j,
                    name=name,
                    value=float(self._show_x(j, value)),
                    value_text=text,
                    conditions=conditions,
                    region=int(regions[j]) if part is not None else 0,
                    contribution=float(contrib[j]) * self._y_std,
                )
            )
        rows.sort(key=lambda r: (-abs(r.contribution), r.feature))

        prediction, label = score, None
        if self.task == "classification":
            if self.link == "logit":
                prediction = float(1.0 / (1.0 + np.exp(-score)))
            label = int(prediction > 0.5)
        return PredictionView(
            task=self.task,
            target_name=self.meta.target_name,
            unit=self.unit,
            baseline=self.baseline * self._y_std + self._y_mean,
            score=score * self._y_std + self._y_mean,
            prediction=(
                prediction * self._y_std + self._y_mean
                if self.task == "regression"
                else prediction
            ),
            label=label,
            contributions=tuple(rows),
        )


# ---------------------------------------------------------------------------
# where the model stands
# ---------------------------------------------------------------------------


def score_table(task: str, y, predictions: dict) -> Scores:
    """Score the GAM, CALM and the black box on the same rows.

    Args:
        task: `"regression"` or `"classification"`.
        y: `(N,)` true targets.
        predictions: `{"gam" | "calm" | "black box": (note, (N,) predictions)}`
            — values for regression, 0/1 labels for classification. A model
            that is not available is left out.

    Returns:
        the `Scores`; the gap is measured on R² (regression) or accuracy.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    names = {"gam": "GAM", "calm": "CALM", "black box": "black box"}

    def metrics(p):
        p = np.asarray(p, dtype=float).reshape(-1)
        if task == "classification":
            return (float(np.mean(p == y)),)
        mse = float(np.mean((y - p) ** 2))
        var = float(np.var(y))
        return (float(np.sqrt(mse)), 1.0 - mse / var if var > 0 else float("nan"))

    rows, last = [], {}
    for key in ("gam", "calm", "black box"):
        if key not in predictions:
            continue
        note, p = predictions[key]
        values = metrics(p)
        rows.append((names[key], note, values))
        last[key] = values[-1]

    gap = None
    if len(last) == 3 and last["black box"] - last["gam"] > 1e-9:
        gap = (last["calm"] - last["gam"]) / (last["black box"] - last["gam"])
    return Scores(
        n_rows=len(y),
        metrics=("accuracy",) if task == "classification" else ("RMSE", "R²"),
        rows=tuple(rows),
        gap=gap,
        gap_metric="accuracy" if task == "classification" else "R²",
    )
