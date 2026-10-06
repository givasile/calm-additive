"""The views of a fitted CALM (`print_summary`, `plot_curves`, …) — the reading contract.

Pins: `baseline + Σ contributions` is the model's score exactly, for both
curve fitters and both tasks; curves are drawn only where their region has
training rows and are centred; importances are read from the fitted curves;
the printed card and ledger carry the rules, the three scores and the sum;
the figures draw every feature kind. Small EBMs, synthetic data.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from calm_additive import CALMClassifier, CALMRegressor  # noqa: E402
from calm_additive.engine import Curve, MaskedFitter  # noqa: E402
from calm_additive.views import score_table  # noqa: E402

FAST_EBM = {"outer_bags": 2, "max_rounds": 200, "learning_rate": 0.1}
N = 2000


def _frame(seed=0):
    """x3's effect flips sign on x2; `kind` and `flag` are categorical."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, 3))
    df = pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
            "kind": pd.Categorical(rng.choice(["red", "green", "blue"], N)),
            "hour": rng.integers(0, 24, N),
        }
    )
    y = (
        X[:, 0] ** 2
        + 2 * np.sin(np.pi / 2 * X[:, 2]) * np.where(X[:, 1] >= 0, 1, -1)
        + (df["kind"] == "red") * 1.0
        + 0.05 * rng.normal(size=N)
    )
    return df, pd.Series(y, name="demand")


@pytest.fixture(scope="module")
def data():
    return _frame()


@pytest.fixture(scope="module")
def reg(data):
    X, y = data
    return CALMRegressor(curve_fitter_params=FAST_EBM).fit(X, y)


@pytest.fixture(scope="module")
def clf(data):
    X, y = data
    labels = pd.Series((y > y.median()).astype(int), name="busy")
    return CALMClassifier(curve_fitter_params=FAST_EBM).fit(X, labels), labels


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fitter", ["ebm", "surrogate"])
def test_contributions_add_up_to_the_regression_score(data, fitter):
    X, y = data
    params = FAST_EBM if fitter == "ebm" else None
    calm = CALMRegressor(curve_fitter=fitter, curve_fitter_params=params).fit(X, y)
    c = calm.contributions(X)
    assert c.shape == (N, 5)
    np.testing.assert_allclose(c.sum(axis=1) + calm.baseline_, calm.predict(X), atol=1e-9)
    # unseen rows too
    X_new, _ = _frame(seed=1)
    np.testing.assert_allclose(
        calm.contributions(X_new).sum(axis=1) + calm.baseline_,
        calm.predict(X_new),
        atol=1e-9,
    )


def test_classifier_contributions_add_up_on_the_log_odds(data, clf):
    X, _ = data
    calm, _ = clf
    score = calm.contributions(X).sum(axis=1) + calm.baseline_
    np.testing.assert_allclose(
        1 / (1 + np.exp(-score)), calm.predict_score(X), atol=1e-9
    )


def test_surrogate_classifier_adds_up_on_the_probability(data, clf):
    X, _ = data
    _, labels = clf
    calm = CALMClassifier(curve_fitter="surrogate").fit(X, labels)
    score = calm.contributions(X).sum(axis=1) + calm.baseline_
    np.testing.assert_allclose(score, calm.predict_score(X), atol=1e-9)


def test_contributions_are_centred_and_the_baseline_is_the_average(data, reg):
    X, _ = data
    c = reg.contributions(X)
    np.testing.assert_allclose(c.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(reg.baseline_, reg.predict(X).mean(), atol=1e-9)


def test_importances_are_the_mean_absolute_contribution(data, reg):
    X, _ = data
    imp = reg.importances()
    np.testing.assert_allclose(imp, np.abs(reg.contributions(X)).mean(axis=0))
    names = list(reg.feature_names_in_)
    # x3 carries the interaction, hour is noise
    assert imp[names.index("x3")] > imp[names.index("x1")] > imp[names.index("hour")]


def test_curves_cover_only_where_the_region_has_rows(data, reg):
    X, _ = data
    curves = reg.curves()
    assert set(curves) == set(reg.feature_names_in_)
    assert [e["region"] for e in curves["x1"]] == ["everywhere"]
    regions = curves["x3"]
    assert len(regions) >= 2  # split on x2
    assert sum(e["n"] for e in regions) == N
    for entry in regions:
        assert "x2" in entry["region"]
        assert entry["x"].min() >= X["x3"].min() and entry["x"].max() <= X["x3"].max()
    # the two sides of x2 = 0 mirror each other
    neg = next(e for e in regions if "<" in e["region"])
    pos = next(e for e in regions if "≥" in e["region"])
    assert np.corrcoef(
        np.interp(np.linspace(-0.9, 0.9, 50), neg["x"], neg["y"]),
        np.interp(np.linspace(-0.9, 0.9, 50), pos["x"], pos["y"]),
    )[0, 1] < -0.9


def test_categorical_curves_are_one_point_per_named_level(reg):
    (entry,) = reg.curves()["kind"]
    assert entry["levels"] == ["blue", "green", "red"]
    assert len(entry["x"]) == len(entry["y"]) == 3
    assert entry["y"][2] > entry["y"][0]  # red is the level that adds


def test_integer_feature_steps_sit_on_the_integers(reg):
    (entry,) = reg.curves()["hour"]
    assert entry["x"].min() == -0.5 and entry["x"].max() == 23.5


def test_nominal_ebm_terms_are_read_as_levels(data):
    X, y = data
    types = ["continuous"] * 3 + ["nominal", "continuous"]
    calm = CALMRegressor(
        max_partitions=0, curve_fitter_params={**FAST_EBM, "feature_types": types}
    ).fit(X, y)
    np.testing.assert_allclose(
        calm.contributions(X).sum(axis=1) + calm.baseline_, calm.predict(X), atol=1e-9
    )


def test_curve_kinds_evaluate_as_documented():
    step = Curve(np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0]), "step")
    np.testing.assert_array_equal(step([-1, 0, 0.5, 1, 2]), [10, 20, 20, 30, 30])
    linear = Curve(np.array([0.0, 1.0]), np.array([0.0, 2.0]), "linear")
    np.testing.assert_allclose(linear([-1, 0.5, 3]), [0, 1, 2])
    levels = Curve(np.array([0.0, 2.0]), np.array([5.0, 7.0]), "levels", other=-1.0)
    np.testing.assert_array_equal(levels([0, 1, 2]), [5, -1, 7])
    np.testing.assert_array_equal(levels.shifted(1.0)([0, 1, 2]), [6, 0, 8])
    np.testing.assert_array_equal(Curve.constant(3.0)([0.0, 9.0]), [3, 3])


# ---------------------------------------------------------------------------
# the printed views
# ---------------------------------------------------------------------------


def test_print_summary_prints_the_card(reg, capsys):
    reg.print_summary()
    out = capsys.readouterr().out
    assert "CALM regressor  ·  target: demand" in out
    assert "5 features" in out and "conditional interaction" in out
    assert "WHERE IT STANDS" not in out
    assert "PARTITIONS" in out and "├─ x2 <" in out and " n = " in out
    # most important first; the split feature names what it depends on
    rows = out.split("FEATURES")[1].split("PARTITIONS")[0].splitlines()
    first = next(line for line in rows if line.strip().startswith("x3"))
    assert first.rstrip().endswith("x2")
    assert all(len(line) <= 76 for line in out.splitlines())


def test_print_summary_with_data_places_calm_between_gam_and_black_box(data, reg, capsys):
    X, y = data
    X_new, y_new = _frame(seed=1)
    reg.print_summary(X_new, y_new)
    out = capsys.readouterr().out
    assert "fitting the GAM reference" in out
    assert f"on the data passed · {N:,} rows" in out
    block = out.split("WHERE IT STANDS")[1].split("FEATURES")[0]
    r2 = {}  # the table rows end with the R² column
    for line in block.splitlines():
        cells = line.split()
        if cells and cells[0] in ("GAM", "CALM", "black") and cells[-1][0].isdigit():
            r2.setdefault(cells[0], float(cells[-1]))
    assert r2["GAM"] < r2["CALM"] <= r2["black"] + 0.05
    assert "of the distance from the GAM to the black box, in R²" in block
    # the GAM is fitted once
    assert hasattr(reg, "gam_")
    reg.print_summary(X_new, y_new)
    assert "fitting the GAM reference" not in capsys.readouterr().out


def test_print_summary_needs_both_x_and_y(data, reg):
    X, _ = data
    with pytest.raises(ValueError, match="both X and y"):
        reg.print_summary(X)


def test_print_prediction_prints_a_ledger_that_adds_up(data, reg, capsys):
    X, _ = data
    reg.print_prediction(X.iloc[7])
    out = capsys.readouterr().out
    assert "PREDICTION" in out and "demand = " in out
    assert "kind = " in out and "because" in out
    row = next(line for line in out.splitlines() if line.strip().startswith("x3 = "))
    assert "x2" in row  # the rule that selected x3's curve
    value = lambda key: float(  # noqa: E731
        next(line for line in out.splitlines() if line.strip().startswith(key))
        .split()[-1]
        .replace("−", "-")
        .replace("+", "")
    )
    assert value("baseline") + value("+ contributions") == pytest.approx(
        value("= demand"), abs=0.02
    )
    assert value("= demand") == pytest.approx(reg.predict(X.iloc[[7]])[0], abs=0.01)


def test_a_row_is_accepted_in_every_shape(data, reg, capsys):
    X, _ = data
    reg.print_prediction(X.iloc[7])
    series = capsys.readouterr().out
    reg.print_prediction(X.iloc[[7]])
    assert capsys.readouterr().out == series
    reg.print_prediction(reg._encode(X.iloc[[7]])[0])
    assert capsys.readouterr().out == series
    with pytest.raises(ValueError, match="one row"):
        reg.print_prediction(X.iloc[:2])


def test_classifier_ledger_ends_with_probability_and_class(data, clf, capsys):
    X, _ = data
    calm, _ = clf
    calm.print_prediction(X.iloc[0])
    out = capsys.readouterr().out
    assert "P(busy = 1) = " in out and "average log-odds" in out
    assert "→ probability" in out and "→ class" in out
    calm.print_summary()
    assert "in log-odds" in capsys.readouterr().out


def test_views_need_a_fitted_model():
    with pytest.raises(RuntimeError, match="not fitted"):
        CALMRegressor().print_summary()


def test_a_fitter_without_curves_predicts_but_cannot_be_explained(data):
    X, y = data

    class Bare(MaskedFitter):
        def fit(self, Xt, y, mask):
            self.mean = float(np.mean(y))
            return self

        def predict(self, Xt, mask):
            return np.full(Xt.shape[0], self.mean)

    calm = CALMRegressor(curve_fitter=Bare()).fit(X, y)
    assert calm.predict(X).shape == (N,)
    with pytest.raises(NotImplementedError, match="curves"):
        calm.print_summary()


def test_refit_resets_the_views(data):
    X, y = data
    calm = CALMRegressor(max_partitions=0, curve_fitter="surrogate").fit(X, y)
    assert len(calm.curves()["x3"]) == 1
    calm.max_partitions = None
    calm.fit(X, y)
    assert len(calm.curves()["x3"]) > 1


# ---------------------------------------------------------------------------
# the scores
# ---------------------------------------------------------------------------


def test_score_table_gap():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    table = score_table(
        "regression",
        y,
        {
            "gam": ("", np.full(4, 1.5)),
            "calm": ("", y + np.array([0.5, -0.5, 0.5, -0.5])),
            "black box": ("", y),
        },
    )
    assert table.metrics == ("RMSE", "R²") and [r[0] for r in table.rows] == [
        "GAM",
        "CALM",
        "black box",
    ]
    assert table.gap == pytest.approx(0.8)
    # no gap to close when the black box is not ahead of the GAM
    flat = score_table(
        "classification",
        np.array([0, 1]),
        {"gam": ("", [0, 1]), "calm": ("", [0, 1]), "black box": ("", [0, 0])},
    )
    assert flat.gap is None and flat.metrics == ("accuracy",)


# ---------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------


def test_plot_draws_one_panel_per_feature(reg):
    fig, axes = reg.plot_curves(show_plot=False)
    assert len(axes) == 5
    titles = [ax.get_title(loc="left") for ax in axes]
    assert titles[0] == "x3"  # most important first
    assert set(titles) == set(reg.feature_names_in_)
    x3 = axes[0]
    assert len(x3.get_lines()) >= 3  # the zero line + one curve per region
    assert reg.plot_curves(features=["x1", "kind"], show_plot=False)[1].__len__() == 2


def test_plot_one_feature_returns_one_axes(reg):
    fig, ax = reg.plot_curves("x3", show_plot=False)
    assert ax.get_title(loc="left") == "x3"
    assert ax.get_ylabel() == "contribution to demand"
    fig2, ax2 = reg.plot_curves(2, show_plot=False)  # by index
    assert ax2.get_title(loc="left") == "x3"


def test_plot_at_a_row_marks_its_contribution(data, reg):
    X, _ = data
    row = X.iloc[7]
    fig, axes = reg.plot_curves(at=row, show_plot=False)
    c = reg.contributions(X.iloc[[7]])[0]
    names = list(reg.feature_names_in_)
    for ax in axes:
        j = names.index(ax.get_title(loc="left"))
        dots = [
            line for line in ax.get_lines()
            if line.get_marker() == "o" and len(line.get_ydata()) == 1
            and line.get_markersize() > 8
        ]
        assert len(dots) == 1
        assert dots[0].get_ydata()[0] == pytest.approx(c[j])


def test_plot_share_y_gives_every_panel_the_same_axis(reg):
    _, axes = reg.plot_curves(share_y=True, show_plot=False)
    assert len({ax.get_ylim() for ax in axes}) == 1


def test_plot_importance_draws_a_segment_per_region(reg):
    fig, ax = reg.plot_importance(show_plot=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "x3"  # the top bar
    n_curves = sum(len(v) for v in reg.curves().values())
    assert len(ax.patches) == n_curves
    widths = sum(p.get_width() for p in ax.patches)
    assert widths == pytest.approx(reg.importances().sum())


def test_plot_prediction_runs_from_the_baseline_to_the_prediction(data, reg):
    X, _ = data
    fig, ax = reg.plot_prediction(X.iloc[7], show_plot=False)
    bars = ax.patches
    assert len(bars) == 5
    start = bars[0].get_x() if bars[0].get_width() >= 0 else bars[0].get_x() + bars[0].get_width()
    assert start == pytest.approx(reg.baseline_)
    total = sum(p.get_width() for p in bars)
    assert reg.baseline_ + total == pytest.approx(reg.predict(X.iloc[[7]])[0])
    # the rest collapse into one bar
    _, ax = reg.plot_prediction(X.iloc[7], max_features=2, show_plot=False)
    assert len(ax.patches) == 3


def test_classifier_figures_draw(data, clf):
    X, _ = data
    calm, _ = clf
    fig, axes = calm.plot_curves(at=X.iloc[0], show_plot=False)
    assert axes[0].get_ylabel() == "contribution (log-odds)"
    calm.plot_importance(show_plot=False)
    _, ax = calm.plot_prediction(X.iloc[0], show_plot=False)
    assert ax.get_xlabel() == "log-odds"


def test_schema_scales_are_applied_to_the_views_only(capsys):
    """A model trained on standardized data is shown in the original units;
    the data accessors stay in the model's units and still add up."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 2))
    y = np.sin(2 * X[:, 0]) * np.where(X[:, 1] > 0, 1, -1) + 0.5 * X[:, 1]
    schema = {
        "feature_names": ["temp", "wind"],
        "target_name": "count",
        "scale_x_list": [{"mean": 20.0, "std": 8.0}, None],
        "scale_y": {"mean": 190.0, "std": 180.0},
    }
    calm = CALMRegressor(curve_fitter="surrogate", schema=schema).fit(X, y)
    np.testing.assert_allclose(
        calm.contributions(X).sum(axis=1) + calm.baseline_, calm.predict(X), atol=1e-9
    )
    (curve, *_) = calm.curves()["temp"]
    assert 20 - 8 * 5 < curve["x"].min() < 0 and curve["x"].max() > 40  # degrees
    calm.print_prediction(X[0])
    out = capsys.readouterr().out
    shown = float(
        next(line for line in out.splitlines() if line.strip().startswith("= count"))
        .split()[-1]
        .replace("−", "-")
    )
    assert shown == pytest.approx(calm.predict(X[:1])[0] * 180.0 + 190.0, abs=0.06)
    assert f"temp = {X[0, 0] * 8 + 20:.4g}" in out
    calm.plot_curves(show_plot=False)


def test_an_explained_model_still_pickles(data):
    import pickle

    X, y = data
    calm = CALMRegressor(curve_fitter="surrogate").fit(X, y)
    calm.curves()  # builds the cached reading (closures over the effect)
    again = pickle.loads(pickle.dumps(calm))
    np.testing.assert_allclose(again.predict(X), calm.predict(X))
    np.testing.assert_allclose(again.importances(), calm.importances())
