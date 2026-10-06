"""Executable CALM (`calm_additive.engine`) — the engine contract.

Pins: the frozen surrogate reproduces the chain's scored R² exactly and
predicts unseen data; the paper's synthetic cases recover their true region
structure; the masked transform routes new points through the leaf rules;
the selection-target axis (teacher vs labels) works; the twins run both
selection modes. Uses analytic teachers only — no heavy deps.
"""

import numpy as np
import pytest

import effector
from calm_additive.engine import (
    CALMClassifier,
    CALMRegressor,
    MaskedFitter,
    MaskedTransform,
    fit_chain,
    fit_partitions,
    register_curve_fitter,
)


def make_X(n=3000, d=3, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, d))


def f1(x):
    """Paper Case 1: x3's effect switches between sine and cosine on sign(x2)."""
    return (
        x[:, 0] ** 2
        + np.log(np.abs(x[:, 1]) + 1e-12)
        + 2 * np.sin(np.pi / 2 * x[:, 2]) * (x[:, 1] >= 0)
        + 2 * np.cos(np.pi / 2 * x[:, 2]) * (x[:, 1] < 0)
    )


def f3(x):
    """Paper Case 3: a general (non-separable) x2-x3 interaction."""
    return x[:, 0] ** 2 + np.log(np.abs(x[:, 1]) + 1e-12) * np.sin(np.pi / 2 * x[:, 2])


def fitted_pdp(X, model):
    m = effector.PDP(X, model, nof_instances="all")
    m.fit(features="all")
    return m


def r2(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)


class LinearMasked(MaskedFitter):
    """Cheap stand-in for the heavy fitters: slopes + per-column offsets."""

    def fit(self, Xt, y, mask):
        Z = np.column_stack([Xt, mask.astype(float)])
        self.beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return self

    def predict(self, Xt, mask):
        return np.column_stack([Xt, mask.astype(float)]) @ self.beta


# ---------------------------------------------------------------------------
# the frozen surrogate is the scored surrogate
# ---------------------------------------------------------------------------


def test_frozen_surrogate_reproduces_the_chain_r2_exactly():
    X = make_X()
    pdp = fitted_pdp(X, f1)
    chain = pdp.select_regions()
    model = fit_chain(chain, curve_fitter="surrogate")
    assert r2(f1(X), model.predict(X)) == pytest.approx(chain.regional_r2, abs=1e-9)
    gam = fit_chain(chain, curve_fitter="surrogate", snapshot=0)
    assert r2(f1(X), gam.predict(X)) == pytest.approx(chain.gam_r2, abs=1e-9)


def test_surrogate_generalizes_to_unseen_points():
    X = make_X()
    pdp = fitted_pdp(X, f1)
    model = fit_chain(pdp.select_regions(), curve_fitter="surrogate")
    X_new = make_X(seed=99)
    assert r2(f1(X_new), model.predict(X_new)) > 0.9


# ---------------------------------------------------------------------------
# the paper's synthetic cases: true regions recovered
# ---------------------------------------------------------------------------


def test_case1_recovers_the_x2_split_on_x3():
    X = make_X()
    chain = fitted_pdp(X, f1).select_regions()
    split_features = {s["feature"] for s in chain.stages}
    assert 2 in split_features, "x3's regime switch must be detected"
    part = chain.final.partitions[2]
    conditioning = {k for leaf in part.leaves for k in leaf.rule.conditions}
    assert conditioning == {1}, "the x3 split must condition on x2"
    # the split point is sign(x2): every leaf boundary sits near 0
    cuts = [
        b
        for leaf in part.leaves
        for b in (leaf.rule[1].lo, leaf.rule[1].hi)
        if np.isfinite(b)
    ]
    assert cuts and max(abs(np.array(cuts))) < 0.15
    # and the regional surrogate is a big jump over the GAM (paper: .74 -> .99)
    assert chain.gam_r2 < 0.85
    assert chain.regional_r2 > 0.95


def test_case3_general_interaction_improves_but_cannot_be_exact():
    X = make_X()
    chain = fitted_pdp(X, f3).select_regions()
    assert chain.regional_r2 > chain.gam_r2 + 0.05


# ---------------------------------------------------------------------------
# masked transform
# ---------------------------------------------------------------------------


def test_masked_transform_routes_new_points_by_the_leaf_rules():
    X = make_X()
    pdp = fitted_pdp(X, f1)
    chain = pdp.select_regions()
    parts = dict(chain.final.partitions)
    tr = MaskedTransform(parts, pdp.dim, pdp.feature_names)
    X_new = make_X(n=500, seed=7)
    Xt, mask = tr.transform(X_new)
    assert Xt.shape == mask.shape == (500, len(tr.new_names))
    for j, part in parts.items():
        cols = tr.foi_2_cols[j]
        # each row is claimed by exactly one leaf of each partitioned feature
        assert (mask[:, cols].sum(axis=1) == 1).all()
        np.testing.assert_allclose(Xt[:, cols].sum(axis=1), X_new[:, j])


def test_fit_partitions_with_a_masked_fitter_beats_the_gam():
    X = make_X()
    y = f1(X)
    pdp = fitted_pdp(X, f1)
    chain = pdp.select_regions()
    model = fit_chain(chain, X, y, curve_fitter=LinearMasked())
    gam = fit_chain(chain, X, y, curve_fitter=LinearMasked(), snapshot=0)
    X_new = make_X(n=1000, seed=3)
    assert r2(f1(X_new), model.predict(X_new)) > r2(f1(X_new), gam.predict(X_new))


# ---------------------------------------------------------------------------
# selection axes
# ---------------------------------------------------------------------------


def test_labels_target_selects_and_fits_against_y():
    X = make_X()
    rng = np.random.default_rng(5)
    y = f1(X) + rng.normal(0, 0.1, len(X))
    pdp = fitted_pdp(X, f1)
    chain = pdp.select_regions(target=y)
    model = fit_chain(chain, y=y, curve_fitter="surrogate", target="y")
    assert r2(y, model.predict(X)) == pytest.approx(chain.regional_r2, abs=1e-9)


# ---------------------------------------------------------------------------
# the twins
# ---------------------------------------------------------------------------


def _register_linear():
    try:
        register_curve_fitter("linear_test", lambda ctx: LinearMasked())
    except Exception:
        pass


def test_twin_r2_gain_selection():
    # surrogate fitter carries the effect curves, so accuracy transfers;
    # a linear masked fitter cannot represent x1^2 / log|x2| by design
    X = make_X()
    y = f1(X)
    calm = CALMRegressor(teacher=f1, curve_fitter="surrogate")
    calm.fit(X, y)
    assert calm.chain_ is not None
    X_new = make_X(n=1000, seed=11)
    assert r2(f1(X_new), calm.predict(X_new)) > 0.9


def test_twin_paper_mode_has_partitions_but_no_chain():
    _register_linear()
    X = make_X()
    y = f1(X)
    calm = CALMRegressor.paper(teacher=f1, curve_fitter="linear_test")
    calm.fit(X, y)
    assert calm.chain_ is None
    assert calm.partitions_
    assert calm.n_conditional_interactions_ >= 1


def test_paper_preset_translates_the_variance_scale_threshold():
    calm = CALMRegressor.paper()
    assert calm.min_heterogeneity_drop == pytest.approx(1 - np.sqrt(0.8))
    partitioner = calm._make_partitioner()
    assert type(partitioner).__name__ == "Best"
    assert partitioner.max_split_levels == 2


# ---------------------------------------------------------------------------
# the hybrid relative gate (T9) flows estimator -> select_regions -> chain
# ---------------------------------------------------------------------------


def test_relative_gain_validation():
    CALMRegressor(min_relative_r2_gain=0.1)._validate_params()
    CALMRegressor(min_relative_r2_gain=None)._validate_params()
    with pytest.raises(ValueError):
        CALMRegressor(min_relative_r2_gain=0.0)._validate_params()
    with pytest.raises(ValueError):
        CALMRegressor(min_relative_r2_gain=1.5)._validate_params()


def test_relative_gain_reaches_the_chain():
    _register_linear()
    X = make_X(n=1500)
    calm = CALMRegressor(
        teacher=f1,
        curve_fitter="linear_test",
        min_relative_r2_gain=0.1,
    )
    calm.fit(X, f1(X))
    assert calm.chain_.relative_gain == 0.1
    assert calm.chain_.to_dict()["relative_gain"] == 0.1


# ---------------------------------------------------------------------------
# the teacher and the inputs
# ---------------------------------------------------------------------------


class CountingTeacher:
    """A non-sklearn teacher that counts its fits."""

    def __init__(self):
        self.nof_fits = 0

    def fit(self, X, y):
        self.nof_fits += 1
        self.is_fitted_ = True

    def forward(self, X):
        return f1(X)


def test_a_fitted_teacher_is_not_refitted():
    X = make_X()
    teacher = CountingTeacher()
    teacher.fit(X, f1(X))
    CALMRegressor(teacher=teacher, curve_fitter="surrogate").fit(X, f1(X))
    assert teacher.nof_fits == 1


def test_an_unfitted_teacher_is_fitted_once():
    X = make_X()
    teacher = CountingTeacher()
    CALMRegressor(teacher=teacher, curve_fitter="surrogate").fit(X, f1(X))
    assert teacher.nof_fits == 1


def test_the_xgb_teacher_reports_itself_fitted():
    from calm_additive.engine import _CALMBase
    from calm_additive.teachers import XGBRegressor

    teacher = XGBRegressor()
    assert not _CALMBase._looks_prefit(teacher)
    X = make_X(200)
    teacher.fit(X, f1(X))
    assert _CALMBase._looks_prefit(teacher)


def test_dataframe_categories_keep_their_codes_at_predict():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(
        {
            "x": rng.uniform(-1, 1, n),
            "z": rng.uniform(-1, 1, n),
            "c": rng.choice(["a", "b", "c"], n),
        }
    )

    def model(X):  # on the encoded matrix: the slope of x depends on the level
        return X[:, 0] * (X[:, 2] - 1.0) + X[:, 1]

    calm = CALMRegressor(teacher=model, curve_fitter="surrogate")
    calm.fit(df, model(calm._ingest(df)[0]))
    full = calm.predict(df)
    keep = (df["c"] != "a").to_numpy()
    # a frame without level "a" must still encode b -> 1, c -> 2
    np.testing.assert_allclose(calm.predict(df[keep]), full[keep])
    np.testing.assert_allclose(calm.predict(df[keep][["c", "z", "x"]]), full[keep])

    with pytest.raises(ValueError, match="level"):
        calm.predict(df.assign(c="unseen"))
    with pytest.raises(ValueError, match="columns"):
        calm.predict(df.drop(columns="z"))
    with pytest.raises(ValueError, match="shape"):
        calm.predict(np.zeros((4, 2)))


def test_schema_fields_override_the_dataframe():
    pd = pytest.importorskip("pandas")
    X = make_X()
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    scale = [{"mean": 10.0, "std": 2.0}, None, None]
    calm = CALMRegressor(
        teacher=f1,
        curve_fitter="surrogate",
        schema={"target_name": "count", "scale_x_list": scale},
    ).fit(df, f1(X))
    meta = calm.effect_.feature_metadata
    assert meta.feature_names == ["a", "b", "c"]
    assert calm.feature_names_in_.tolist() == ["a", "b", "c"]
    assert calm.n_features_in_ == 3
    assert meta.target_name == "count"
    assert meta.scale_x_list == scale


def test_classification_needs_probabilities_and_a_binary_target():
    X = make_X(500)
    y = (f1(X) > 0).astype(int)

    class LabelsOnly:
        is_fitted_ = True

        def predict(self, X):
            return (f1(X) > 0).astype(int)

    with pytest.raises(ValueError, match="probability"):
        CALMClassifier(teacher=LabelsOnly(), curve_fitter="surrogate").fit(X, y)
    proba = lambda X: 1.0 / (1.0 + np.exp(-f1(X)))  # noqa: E731
    for bad in (y + 1, np.where(y == 1, "yes", "no"), np.zeros_like(y)):
        with pytest.raises(ValueError, match="binary"):
            CALMClassifier(teacher=proba, curve_fitter="surrogate").fit(X, bad)
    with pytest.raises(ValueError, match="rows"):
        CALMClassifier(teacher=proba, curve_fitter="surrogate").fit(X, y[:-1])
    CALMClassifier(teacher=proba, curve_fitter="surrogate").fit(X, y.astype(bool))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_missing_and_infinite_values_are_rejected(bad):
    X = make_X(500)
    y = f1(X)
    calm = CALMRegressor(teacher=f1, curve_fitter="surrogate").fit(X, y)
    Xb = X.copy()
    Xb[3, 1] = bad
    with pytest.raises(ValueError, match="'x_1' contains missing or infinite"):
        calm.predict(Xb)
    with pytest.raises(ValueError, match="'x_1' contains missing or infinite"):
        CALMRegressor(teacher=f1, curve_fitter="surrogate").fit(Xb, y)
    yb = y.copy()
    yb[0] = bad
    with pytest.raises(ValueError, match="target contains missing or infinite"):
        CALMRegressor(teacher=f1, curve_fitter="surrogate").fit(X, yb)


# ---------------------------------------------------------------------------
# the slots and their *_params
# ---------------------------------------------------------------------------


def test_the_flat_knobs_reach_the_partitioner():
    calm = CALMRegressor(
        max_partition_depth=1,
        min_heterogeneity_drop=0.2,
        partitioner_params={"min_samples_leaf": 50},
    )
    partitioner = calm._make_partitioner()
    assert partitioner.max_split_levels == 1
    assert partitioner.heter_pcg_drop_thres == 0.2
    assert partitioner.min_points_per_subregion == 50


def test_partitioner_conflicts_raise():
    from effector.space_partitioning import Best

    assert isinstance(CALMRegressor(partitioner=Best())._make_partitioner(), Best)
    with pytest.raises(ValueError, match="max_partition_depth"):
        CALMRegressor(partitioner=Best(), max_partition_depth=1)._make_partitioner()
    with pytest.raises(ValueError, match="max_partition_depth"):
        CALMRegressor(partitioner_params={"max_depth": 1})._make_partitioner()


def test_params_with_an_object_raise():
    X = make_X(200)
    with pytest.raises(ValueError, match="teacher_params"):
        CALMRegressor(teacher=f1, teacher_params={"a": 1}).fit(X, f1(X))
    with pytest.raises(ValueError, match="curve_fitter_params"):
        CALMRegressor(
            teacher=f1, curve_fitter=LinearMasked(), curve_fitter_params={"a": 1}
        ).fit(X, f1(X))
    with pytest.raises(ValueError, match="selection"):
        CALMRegressor(teacher=f1, selection="greedy").fit(X, f1(X))


def test_params_reach_the_named_components():
    from calm_additive.engine import _TEACHERS, _build
    from calm_additive.fitters import EBMRegressor

    teacher = _build(_TEACHERS["xgb"], "regression", {"n_estimators": 7}, 3)
    assert teacher.model.n_estimators == 7
    assert teacher.model.random_state == 3
    assert teacher.model.learning_rate == 0.1
    assert EBMRegressor(max_bins=64).model.max_bins == 64
    assert EBMRegressor(max_bins=64).model.interactions == 0
    with pytest.raises(ValueError, match="interactions"):
        EBMRegressor(interactions=3)


def test_effect_params_are_routed_to_effector():
    X = make_X(600)
    calm = CALMRegressor(
        teacher=f1,
        curve_fitter="surrogate",
        effect_params={"nof_instances": 300, "centering": True},
    ).fit(X, f1(X))
    assert calm.effect_.data.shape[0] == 300
    with pytest.raises(ValueError, match="no setting"):
        CALMRegressor(
            teacher=f1, curve_fitter="surrogate", effect_params={"nope": 1}
        ).fit(X, f1(X))


# ---------------------------------------------------------------------------
# the caps: max_conditional_interactions (K) and max_partitions
# ---------------------------------------------------------------------------


def f_two_switches(x):
    """Two separate conditional interactions: x0 | x1 and x2 | x3."""
    return 2 * x[:, 0] * np.sign(x[:, 1]) + x[:, 2] * np.sign(x[:, 3]) + x[:, 1]


def _fit_two_switches(**kwargs):
    X = make_X(d=4)
    calm = CALMRegressor(
        teacher=f_two_switches,
        curve_fitter="surrogate",
        partition_features="all",
        max_partition_depth=1,
        **kwargs,
    )
    return calm.fit(X, f_two_switches(X))


def test_the_default_cap_is_ten_conditional_interactions():
    calm = CALMRegressor()
    assert calm.max_conditional_interactions == 10
    assert calm.max_partitions is None


def test_max_conditional_interactions_stops_at_a_prefix_of_the_chain():
    full = _fit_two_switches(max_conditional_interactions=None)
    assert full.n_conditional_interactions_ >= 2

    capped = _fit_two_switches(max_conditional_interactions=1)
    assert capped.n_conditional_interactions_ == 1
    assert capped.chain_.stages == full.chain_.stages[:1]
    assert "max_conditional_interactions" in {
        sk["reason"] for sk in capped.chain_.skipped
    }

    gam = _fit_two_switches(max_conditional_interactions=0)
    assert gam.partitions_ == {} and gam.n_conditional_interactions_ == 0


def test_max_partitions_caps_the_split_features():
    full = _fit_two_switches(max_conditional_interactions=None)
    assert len(full.partitions_) >= 2
    capped = _fit_two_switches(max_partitions=1)
    assert len(capped.partitions_) == 1
    assert capped.chain_.stages == full.chain_.stages[:1]
    assert "max_partitions" in {sk["reason"] for sk in capped.chain_.skipped}


def test_a_cap_cannot_be_combined_with_selection_all():
    with pytest.raises(ValueError, match="max_conditional_interactions=None"):
        CALMRegressor(selection="all")._validate_params()
    with pytest.raises(ValueError, match="max_partitions=None"):
        CALMRegressor(
            selection="all", max_conditional_interactions=None, max_partitions=3
        )._validate_params()
    CALMRegressor(selection="all", max_conditional_interactions=None)._validate_params()
    assert CALMRegressor.paper().max_conditional_interactions is None


@pytest.mark.parametrize("bad", [-1, 2.5, "3", True])
def test_cap_validation(bad):
    with pytest.raises(ValueError, match="max_conditional_interactions"):
        CALMRegressor(max_conditional_interactions=bad)._validate_params()
    with pytest.raises(ValueError, match="max_partitions"):
        CALMRegressor(max_partitions=bad)._validate_params()


# ---------------------------------------------------------------------------
# verbose: the stage bar
# ---------------------------------------------------------------------------


def test_verbose_shows_a_stage_bar_and_silent_shows_nothing(capfd):
    X = make_X()
    CALMRegressor(teacher=f1, curve_fitter="surrogate").fit(X, f1(X))
    assert capfd.readouterr().err == ""
    CALMRegressor(teacher=f1, curve_fitter="surrogate", verbose=1).fit(X, f1(X))
    err = capfd.readouterr().err
    assert "CALM · fitted" in err and "5/5" in err
    assert "teacher (function)" in err and "curves (surrogate)" in err
    CALMRegressor.paper(
        teacher=f1, curve_fitter="surrogate", verbose=1
    ).fit(X, f1(X))
    assert "4/4" in capfd.readouterr().err


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_verbose_validation(bad):
    with pytest.raises(ValueError, match="verbose"):
        CALMRegressor(verbose=bad)._validate_params()
