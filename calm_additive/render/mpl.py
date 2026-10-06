"""The figures, drawn with matplotlib.

Each function takes a view from `calm_additive.views` and draws it;
nothing about the model is computed here. Colours come from effector's
active theme (`effector.set_theme`) and are set on the artists themselves,
so the figures look the same whatever matplotlib's global style is.

Layout is in inches, placed by hand: a panel is a curve axes, a density
strip under it (one row per region) and, for a feature with a partition, the
legend under that — heights that depend on the number of regions, which
`tight_layout` cannot balance across a grid.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D

from effector import theme
from effector.visualization import _categorical_axis, _finalize

from calm_additive.render._format import clip, decimals, fixed, num, plural, signed

# panel geometry, inches
_PANEL_W, _CURVE_H = 4.3, 2.15  # a grid panel: full width, curve height
_SINGLE_W, _SINGLE_H = 7.6, 3.5  # one feature on its own
_LEFT, _RIGHT = 0.72, 0.28  # room for the y ticks; gap to the next panel
_TITLE_H, _STRIP_ROW, _STRIP_GAP = 0.34, 0.13, 0.05
_TICKS_H, _LEGEND_ROW, _BOTTOM = 0.3, 0.19, 0.24


# ---------------------------------------------------------------------------
# the look
# ---------------------------------------------------------------------------


def _ink(t):
    """The text and chrome colours of the active theme."""
    rc = t.rcparams
    return {
        "surface": rc["axes.facecolor"],
        "title": rc["axes.titlecolor"],
        "label": rc["axes.labelcolor"],
        "tick": rc["xtick.color"],
        "grid": rc["grid.color"],
        "spine": rc["axes.edgecolor"],
    }


def _chrome(ax, ink, grid="both"):
    """Recessive axes: no box, hairline grid, muted ticks."""
    ax.set_facecolor(ink["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["spine"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=ink["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink["tick"], labelsize=8, length=3, width=0.8)


def _halo(ink):
    """A surface-coloured outline that keeps a label legible over marks."""
    return [patheffects.withStroke(linewidth=2.5, foreground=ink["surface"])]


def _region_colors(t, feature):
    """One colour per region, in the theme's fixed order; the theme's main
    colour for a feature with a single curve."""
    if not feature.is_split:
        return [t.MEAN]
    return [t.CAT[i % len(t.CAT)] for i in range(len(feature.regions))]


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------


def _level_step(feature) -> float:
    """The distance between neighbouring levels on a categorical axis."""
    p = feature.level_positions
    return float(np.min(np.diff(p))) if len(p) > 1 else 1.0


def _dodge(feature, i):
    """Sideways offset of region `i`'s marks on a categorical axis."""
    n = len(feature.regions)
    return 0.0 if n == 1 else (i - (n - 1) / 2) * (0.62 / n) * _level_step(feature)


def _draw_curves(ax, feature, colors, ink, active=None):
    """The feature's curves on `ax`. With `active` (a region index, or -1)
    only that region keeps its colour; the others recede."""
    for i, (region, color) in enumerate(zip(feature.regions, colors)):
        on = active is None or i == active
        style = dict(color=color, alpha=1.0) if on else dict(color=ink["tick"], alpha=0.45)
        z = 3 if on else 2
        if feature.is_categorical:
            x = region.x + _dodge(feature, i)
            ax.vlines(x, 0, region.y, linewidth=1.4, zorder=z, **{**style, "alpha": style["alpha"] * 0.55})
            ax.plot(
                x, region.y, "o", markersize=6.5, markeredgecolor=ink["surface"],
                markeredgewidth=1.2, zorder=z + 0.5, **style,
            )
        elif len(region.x) == 1:  # the region has a single value of the feature
            ax.plot(
                region.x, region.y, "o", markersize=6.5,
                markeredgecolor=ink["surface"], markeredgewidth=1.2, zorder=z, **style,
            )
        else:
            ax.plot(
                region.x, region.y, linewidth=2.0 if on else 1.2,
                solid_joinstyle="round", solid_capstyle="round", zorder=z, **style,
            )


def _draw_density(ax, feature, colors, ink, active=None):
    """Where the training rows are: one row of bars per region, all rows on
    the scale of the feature's fullest bin."""
    n = len(feature.regions)
    top = max((r.density.max() for r in feature.regions if len(r.density)), default=0)
    for i, (region, color) in enumerate(zip(feature.regions, colors)):
        on = active is None or i == active
        base = n - 1 - i
        heights = 0.86 * region.density / top if top > 0 else region.density * 0.0
        if feature.is_categorical:
            width = 0.6 * _level_step(feature)
            left = feature.level_positions - width / 2
        else:
            left, width = feature.bins[:-1], np.diff(feature.bins)
        ax.bar(
            left, heights, width=width, bottom=base, align="edge", linewidth=0,
            color=color if on else ink["tick"], alpha=0.55 if on else 0.22, zorder=2,
        )
    ax.set_ylim(0, n)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)


def _draw_legend(ax, feature, colors, ink, active=None, width_chars=58):
    """The regions under the panel: colour key, rule, number of rows."""
    n = len(feature.regions)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")
    for i, (region, color) in enumerate(zip(feature.regions, colors)):
        on = active is None or i == active
        y = n - 0.5 - i
        ax.add_line(
            Line2D(
                [0.0, 0.055], [y, y], linewidth=2.0, solid_capstyle="round",
                color=color if on else ink["tick"], alpha=1.0 if on else 0.45,
            )
        )
        ax.text(
            0.075, y, clip(region.label, width_chars), va="center", ha="left",
            fontsize=7.8, color=ink["label"] if on else ink["tick"],
        )
        ax.text(
            1.0, y, f"n = {region.n:,}", va="center", ha="right",
            fontsize=7.2, color=ink["tick"],
        )


def _mark_row(ax, feature, colors, row, ink):
    """One row on its curve: a dot at the feature's value, labelled with the
    contribution. Called once the axes limits are final: the label goes to
    the side of the dot that has room."""
    if feature.is_categorical:
        if row.value_text not in feature.level_names:
            return  # a level the training data did not have
        x = feature.level_positions[feature.level_names.index(row.value_text)]
        x += _dodge(feature, row.region) if row.region >= 0 else 0.0
    else:
        x = row.value
    color = colors[row.region] if row.region >= 0 else ink["title"]
    ax.axvline(x, color=ink["spine"], linewidth=0.8, zorder=1)
    ax.plot(
        [x], [row.contribution], "o", markersize=8.5, color=color,
        markeredgecolor=ink["surface"], markeredgewidth=1.6, zorder=6,
    )
    fx, fy = ax.transLimits.transform((x, row.contribution))
    right, up = fx < 0.72, fy < 0.8
    ax.annotate(
        signed(row.contribution), (x, row.contribution),
        xytext=(8 if right else -8, 7 if up else -7), textcoords="offset points",
        ha="left" if right else "right", va="bottom" if up else "top",
        fontsize=8.5, fontweight="bold", color=ink["title"], zorder=7,
        path_effects=_halo(ink),
    )


def _unit_label(view) -> str:
    if view.task == "regression":
        return f"contribution to {view.target_name}"
    return f"contribution ({view.unit})"


def plot_curves(view, features, *, at=None, ncols=3, share_y=False, single=False, show_plot=True):
    """One panel per feature, one curve per region.

    Args:
        view: a `ModelView`.
        features: indices of the features to draw, in panel order.
        at: a `PredictionView` — its row is marked on every panel.
        ncols: panels per row.
        share_y: one y-axis for every panel.
        single: `features` holds one feature, drawn large.
        show_plot: show and return `None`, or return `(fig, axes)`.
    """
    t = theme.active()
    ink = _ink(t)
    feats = [view.features[j] for j in features]
    rows_at = {} if at is None else {c.feature: c for c in at.contributions}

    if single:
        ncols, panel_w, curve_h, chars = 1, _SINGLE_W, _SINGLE_H, 110
    else:
        ncols = max(1, min(ncols, len(feats)))
        panel_w, curve_h, chars = _PANEL_W, _CURVE_H, 52
    nrows = int(np.ceil(len(feats) / ncols))

    def strip_h(f):
        return _STRIP_ROW * len(f.regions)

    def ticks_h(f):
        # level names may be rotated: leave them room
        long_names = f.level_labels is not None and (
            len(f.levels) > 6 or max(len(n) for n in f.level_labels) > 6
        )
        return _TICKS_H + (0.42 if long_names else 0.0)

    def legend_h(f):
        return _LEGEND_ROW * len(f.regions) + 0.06 if f.is_split else 0.0

    grid = [feats[r * ncols : (r + 1) * ncols] for r in range(nrows)]
    row_h = [
        _TITLE_H + curve_h + _STRIP_GAP
        + max(strip_h(f) + ticks_h(f) + legend_h(f) for f in row) + _BOTTOM
        for row in grid
    ]
    head = 0.42 if at is not None else 0.08
    fig_w, fig_h = ncols * panel_w, head + sum(row_h)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=ink["surface"])

    def place(left, top, width, height, **kwargs):
        """An axes from its top-left corner, in inches from the figure's top."""
        return fig.add_axes(
            [left / fig_w, 1 - (top + height) / fig_h, width / fig_w, height / fig_h],
            **kwargs,
        )

    if at is not None:
        fig.text(
            _LEFT / fig_w, 1 - 0.24 / fig_h, _headline(at), fontsize=9.5,
            color=ink["label"], ha="left", va="center",
        )

    axes, marks = [], []
    top = head
    for r, row in enumerate(grid):
        for c, f in enumerate(row):
            left, width = c * panel_w + _LEFT, panel_w - _LEFT - _RIGHT
            colors = _region_colors(t, f)
            here = rows_at.get(f.index)
            active = None if here is None else here.region

            ax = place(left, top + _TITLE_H, width, curve_h)
            _chrome(ax, ink)
            ax.axhline(0, color=t.REF, linewidth=0.8, zorder=1)
            _draw_curves(ax, f, colors, ink, active)
            ax.margins(x=0.04, y=0.1)
            if here is not None:
                marks.append((ax, f, colors, here))
            ax.set_title(
                clip(f.name, 34), loc="left", fontsize=10.5, fontweight="bold",
                color=ink["title"], pad=5,
            )
            tag = (
                f"{f.name} = {here.value_text}"
                if here is not None
                else f"importance {num(f.importance)}"
            )
            ax.text(
                1.0, 1.025, clip(tag, 30), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, color=t.TAG,
            )
            if c == 0:
                ax.set_ylabel(_unit_label(view), fontsize=8.5, color=ink["label"])
            ax.tick_params(labelbottom=False, bottom=False)

            y = top + _TITLE_H + curve_h + _STRIP_GAP
            sax = place(left, y, width, strip_h(f), sharex=ax)
            _chrome(sax, ink, grid=None)
            _draw_density(sax, f, colors, ink, active)
            if f.is_categorical:
                labels = None if f.level_labels is None else list(f.level_labels)
                _categorical_axis(sax, f.level_positions, labels, f.kind)
                sax.tick_params(labelsize=8)
                pad = 0.6 * _level_step(f)
                ax.set_xlim(f.level_positions[0] - pad, f.level_positions[-1] + pad)
            if f.is_split:
                lax = place(left, y + strip_h(f) + ticks_h(f), width, legend_h(f) - 0.06)
                _draw_legend(lax, f, colors, ink, active, chars)
            axes.append(ax)
        top += row_h[r]

    if share_y and axes:
        lo = min(ax.get_ylim()[0] for ax in axes)
        hi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(lo, hi)
    for mark in marks:
        _mark_row(*mark, ink)
    return _finalize(fig, axes[0] if single else axes, show_plot)


def _headline(at) -> str:
    """The prediction in one line, over the panels."""
    if at.task == "classification":
        lead = f"P({at.target_name} = 1) = {at.prediction:.3f}"
        if at.unit == "log-odds":
            lead += f"   ·   log-odds {num(at.score)}"
    else:
        lead = f"{at.target_name} = {num(at.prediction)}"
    total = sum(c.contribution for c in at.contributions)
    return (
        f"{lead}   =   baseline {num(at.baseline)}  {signed(total)}"
        "  (the marked contributions, summed)"
    )


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------


def plot_importance(view, *, max_features=15, show_plot=True):
    """The features by importance, one bar each; the bar of a feature with a
    partition is cut into one segment per region.

    Args:
        view: a `ModelView`.
        max_features: bars drawn, most important first.
        show_plot: show and return `None`, or return `(fig, ax)`.
    """
    t = theme.active()
    ink = _ink(t)
    shown = [view.features[j] for j in view.order[:max_features]]
    rest = len(view.order) - len(shown)
    k = len(shown)

    fig, ax = plt.subplots(
        figsize=(6.6, 0.36 * k + 1.25), facecolor=ink["surface"]
    )
    _chrome(ax, ink, grid="x")
    ax.spines["bottom"].set_visible(False)
    top = max((f.importance for f in shown), default=0.0)
    places = decimals([top])
    for i, f in enumerate(shown):
        y, left = k - 1 - i, 0.0
        for region in f.regions:
            # surface-coloured edges are the gaps between a bar's segments
            ax.barh(
                y, region.share, left=left, height=0.56, color=t.MEAN,
                edgecolor=ink["surface"], linewidth=1.6, zorder=3,
            )
            left += region.share
        value = ax.text(
            f.importance + 0.012 * top, y, fixed(f.importance, places),
            va="center", ha="left", fontsize=8, color=ink["label"], zorder=4,
        )
        if f.is_split:
            ax.annotate(
                f"{len(f.regions)} regions", xy=(1, 0.5), xycoords=value,
                xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7.2, color=t.TAG,
            )
    ax.set_yticks(range(k))
    ax.set_yticklabels([clip(f.name, 26) for f in reversed(shown)], fontsize=9)
    ax.tick_params(axis="y", colors=ink["label"], length=0)
    ax.set_ylim(-0.6, k - 0.4)
    ax.set_xlim(0, top * 1.24 if top > 0 else 1)
    unit = f"to {view.target_name}" if view.task == "regression" else f"({view.unit})"
    ax.set_xlabel(f"mean |contribution| {unit}", fontsize=8.5, color=ink["label"])
    ax.set_title(
        "Feature importance", loc="left", fontsize=10.5, fontweight="bold",
        color=ink["title"], pad=8,
    )
    if rest > 0:
        ax.text(
            1.0, 1.02, f"… {plural(rest, 'more feature')} not shown",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=t.TAG,
        )
    fig.tight_layout()
    return _finalize(fig, ax, show_plot)


# ---------------------------------------------------------------------------
# one prediction
# ---------------------------------------------------------------------------


def plot_prediction(view, *, max_features=10, show_plot=True):
    """One prediction as a waterfall: the baseline, one bar per feature
    (largest |contribution| first), the prediction.

    Args:
        view: a `PredictionView`.
        max_features: bars drawn; the rest are summed in one bar.
        show_plot: show and return `None`, or return `(fig, ax)`.
    """
    t = theme.active()
    ink = _ink(t)
    rows = [
        (f"{c.name} = {c.value_text}", ", ".join(c.conditions), c.contribution)
        for c in view.contributions[:max_features]
    ]
    rest = view.contributions[max_features:]
    if rest:
        rows.append(
            (plural(len(rest), "other feature"), "", sum(c.contribution for c in rest))
        )
    k = len(rows)

    left_in, fig_w = 2.75, 7.8
    fig_h = 0.47 * k + 1.75
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=ink["surface"])
    ax = fig.add_axes(
        [left_in / fig_w, 0.62 / fig_h, (fig_w - left_in - 0.35) / fig_w, (fig_h - 1.3) / fig_h]
    )
    _chrome(ax, ink, grid="x")
    ax.spines["left"].set_visible(False)

    places = decimals([value for _, _, value in rows])
    running = view.baseline
    ends = [running]
    for i, (label, because, value) in enumerate(rows):
        y = k - 1 - i
        color = t.MEAN if value >= 0 else t.ARROW
        ax.barh(y, value, left=running, height=0.5, color=color, linewidth=0, zorder=3)
        # the running total carries on to the next bar
        ax.plot(
            [running + value] * 2, [y - 0.25, y - 0.75], color=ink["spine"],
            linewidth=0.8, zorder=2,
        )
        end = running + value
        ax.annotate(
            fixed(value, places, sign=True), (end, y),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points", va="center",
            ha="left" if value >= 0 else "right", fontsize=8, color=ink["label"],
            zorder=4,
        )
        trans = ax.get_yaxis_transform()
        if because:
            ax.text(-0.025, y + 0.04, clip(label, 34), transform=trans, ha="right",
                    va="bottom", fontsize=8.8, color=ink["label"])
            ax.text(-0.025, y - 0.06, clip(because, 46), transform=trans, ha="right",
                    va="top", fontsize=7, color=t.TAG)
        else:
            ax.text(-0.025, y, clip(label, 34), transform=trans, ha="right",
                    va="center", fontsize=8.8, color=ink["label"])
        running = end
        ends.append(running)

    ax.axvline(view.baseline, color=t.REF, linewidth=0.8, zorder=1)
    ax.plot(
        [view.score], [-1], "o", markersize=8.5, color=ink["title"],
        markeredgecolor=ink["surface"], markeredgewidth=1.6, zorder=5, clip_on=False,
    )
    span = (max(ends) - min(ends)) or 1.0
    ax.set_xlim(min(ends) - 0.14 * span, max(ends) + 0.14 * span)
    ax.set_ylim(-1.45, k - 0.4)
    ax.set_yticks([])
    trans = ax.get_yaxis_transform()
    ax.text(-0.025, -1, _prediction_label(view), transform=trans, ha="right",
            va="center", fontsize=8.8, fontweight="bold", color=ink["title"])
    ax.annotate(
        f"baseline {num(view.baseline)}", (view.baseline, k - 0.4), xytext=(0, 4),
        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=t.TAG,
    )
    xlabel = view.target_name if view.task == "regression" else view.unit
    ax.set_xlabel(xlabel, fontsize=8.5, color=ink["label"])
    fig.text(
        0.3 / fig_w, 1 - 0.3 / fig_h, _prediction_title(view), fontsize=10.5,
        fontweight="bold", color=ink["title"], ha="left", va="center",
    )
    return _finalize(fig, ax, show_plot)


def _prediction_label(view) -> str:
    return f"{view.unit if view.task == 'classification' else 'prediction'} = {num(view.score)}"


def _prediction_title(view) -> str:
    if view.task == "classification":
        return f"P({view.target_name} = 1) = {view.prediction:.3f}   →   class {view.label}"
    return f"{view.target_name} = {num(view.prediction)}"
