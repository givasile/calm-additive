"""The printed views: the model card and one prediction's ledger.

Each function takes a view from `calm_additive.views` and returns the
text; nothing is computed here. The layout follows effector's ledger
(`chain_.show()`): sections under a rule, every table `INDENT + TABLE` wide.
"""

from __future__ import annotations

from calm_additive.render._format import clip as _clip
from calm_additive.render._format import decimals as _decimals
from calm_additive.render._format import fixed as _fixed
from calm_additive.render._format import num as _num
from calm_additive.render._format import plural as _plural

INDENT, TABLE = 4, 70
_PAD = " " * INDENT


class _Page:
    """Lines under effector's section layout."""

    def __init__(self):
        self.lines = []

    def add(self, line=""):
        self.lines.append(line.rstrip())

    def row(self, text=""):
        self.add(_PAD + text)

    def rule(self, ch="─", width=TABLE, indent=INDENT):
        self.add(" " * indent + ch * width)

    def section(self, name, note=""):
        self.add()
        head = " " * (INDENT - 2) + name
        if note:
            head += " " * max(2, INDENT + TABLE - len(head) - len(note)) + note
        self.add(head)
        self.rule("─", TABLE + 2, INDENT - 2)

    def title(self, text):
        self.add()
        self.rule("═", TABLE + 2, INDENT - 2)
        self.add(" " * (INDENT - 2) + text)
        self.rule("═", TABLE + 2, INDENT - 2)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------------------
# the model card
# ---------------------------------------------------------------------------


def _where_it_stands(page, scores):
    page.section("WHERE IT STANDS", f"on the data passed · {scores.n_rows:,} rows")
    # 12 + 34 + 12 per metric
    head = f"{'':<12}{'':<34}" + "".join(f"{m:>12}" for m in scores.metrics)
    page.row(head)
    # one format per column: errors at the size of the largest, ratios at 3
    places = [
        _decimals([values[i] for _, _, values in scores.rows]) if m == "RMSE" else 3
        for i, m in enumerate(scores.metrics)
    ]
    for name, note, values in scores.rows:
        cells = "".join(f"{_fixed(v, d):>12}" for v, d in zip(values, places))
        page.row(f"{name:<12}{_clip(note, 33):<34}{cells}")

    gap = scores.gap
    if gap is None:
        if len(scores.rows) == 3:
            page.row()
            page.row(
                f"the black box is not ahead of the GAM in {scores.gap_metric}: "
                "no gap to close"
            )
        return
    page.row()
    if gap > 1:
        page.row(f"CALM is above the black box in {scores.gap_metric}")
    elif gap < 0:
        page.row(f"CALM is below the GAM in {scores.gap_metric}")
    else:
        width = 44
        at = int(round(gap * (width - 1)))
        track = ["━"] * at + ["─"] * (width - at)
        track[0], track[-1] = "●", "○"
        track[at] = "●"
        page.row("GAM " + "".join(track) + " black box")
        page.row(" " * max(0, min(4 + at - 2, 4 + width - 4)) + "CALM")
        page.row(
            f"CALM covers {gap:.0%} of the distance from the GAM to the "
            f"black box, in {scores.gap_metric}"
        )


def model_card(view, max_features: int = 20) -> str:
    """The model on one screen: its size, where it stands (when scored), the
    features by importance, and each partition as a tree.

    Args:
        view: a `ModelView`.
        max_features: rows of the feature table; the rest are summed up.

    Returns:
        the card as text, ending with a newline.
    """
    page = _Page()
    kind = "classifier" if view.task == "classification" else "regressor"
    page.title(f"CALM {kind}  ·  target: {view.target_name}")

    page.section("MODEL")
    rows = [
        ("trained on", f"{view.n_rows:,} rows · {_plural(len(view.features), 'feature')}"),
        (
            "structure",
            f"{_plural(view.n_curves, 'curve')} · "
            f"{_plural(view.n_partitions, 'partition')} · "
            f"{_plural(view.n_conditional_interactions, 'conditional interaction')}",
        ),
        ("baseline", f"{_num(view.baseline)}  (the average {_score_word(view)})"),
    ]
    for key, value in rows:
        page.row(f"{key:<14}{value}")

    if view.scores is not None:
        _where_it_stands(page, view.scores)

    # 14 + 10 + 2 + 12 + 8 + 2 + 22 = 70
    page.section("FEATURES", f"importance = mean |contribution|, in {view.unit}")
    page.row(
        f"{'feature':<14}{'importance':>10}  {'':<12}{'curves':>8}  {'depends on'}"
    )
    page.rule()
    total = sum(f.importance for f in view.features)
    top = max((f.importance for f in view.features), default=0.0)
    places = _decimals([top])
    shown = view.order[:max_features]
    for j in shown:
        f = view.features[j]
        bar = "█" * (int(round(12 * f.importance / top)) if top > 0 else 0)
        depends = _clip(", ".join(f.conditioned_on), 22) if f.is_split else "—"
        page.row(
            f"{_clip(f.name, 13):<14}{_fixed(f.importance, places):>10}  {bar:<12}"
            f"{len(f.regions):>8}  {depends}"
        )
    rest = view.order[max_features:]
    if rest:
        page.rule()
        left = sum(view.features[j].importance for j in rest)
        share = left / total if total > 0 else 0.0
        page.row(
            f"… {_plural(len(rest), 'more feature')}, "
            f"{share:.0%} of the total importance"
        )

    split = [view.features[j] for j in view.order if view.features[j].is_split]
    if split:
        page.section("PARTITIONS", "one curve per leaf")
        for i, f in enumerate(split):
            if i:
                page.row()
            for line in f.tree:
                page.row(line)
    return page.text()


def _score_word(view) -> str:
    if view.task == "regression":
        return "prediction"
    return "log-odds" if view.unit == "log-odds" else "probability"


# ---------------------------------------------------------------------------
# one prediction
# ---------------------------------------------------------------------------


def _wrap_conditions(conditions, width: int) -> list:
    """The rule over as few lines as fit: conditions joined by ", "."""
    lines, current = [], ""
    for cond in conditions:
        cond = _clip(cond, width)
        candidate = f"{current}, {cond}" if current else cond
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current + ",")
            current = cond
    if current:
        lines.append(current)
    return lines or [""]


def _diverging_bar(value: float, largest: float, half: int = 7) -> str:
    """`value` as a bar growing left (negative) or right from a centre line."""
    n = int(round(half * abs(value) / largest)) if largest > 0 else 0
    if value < 0:
        return " " * (half - n) + "█" * n + "│"
    return " " * half + "│" + "█" * n


def prediction(view, max_features: int = 10) -> str:
    """One prediction as a ledger: per feature its value, the rule that
    selected its curve, and what it adds to the baseline.

    Args:
        view: a `PredictionView`.
        max_features: rows shown, largest |contribution| first; the rest are
            summed in one row.

    Returns:
        the ledger as text, ending with a newline.
    """
    page = _Page()
    if view.task == "classification":
        headline = f"P({view.target_name} = 1) = {view.prediction:.3f}"
    else:
        headline = f"{view.target_name} = {_num(view.prediction)}"
    page.section("PREDICTION", headline)

    # 20 + 22 + 12 + 1 + 15 = 70
    page.row(f"{'feature = value':<20}{'because':<22}{'contribution':>12}")
    page.rule()
    rows = view.contributions[:max_features]
    rest = view.contributions[max_features:]
    total = sum(r.contribution for r in view.contributions)
    largest = max((abs(r.contribution) for r in view.contributions), default=0.0)
    places = _decimals([largest, view.baseline, view.score])
    for r in rows:
        because = _wrap_conditions(r.conditions, 21)
        head = _clip(f"{r.name} = {r.value_text}", 19)
        page.row(
            f"{head:<20}{because[0]:<22}{_fixed(r.contribution, places, sign=True):>12} "
            f"{_diverging_bar(r.contribution, largest)}"
        )
        for extra in because[1:]:
            page.row(f"{'':<20}{extra}")
    if rest:
        other = sum(r.contribution for r in rest)
        label = f"… {_plural(len(rest), 'more feature')}"
        page.row(
            f"{label:<42}{_fixed(other, places, sign=True):>12} "
            f"{_diverging_bar(other, largest)}"
        )
    page.rule()

    page.row(
        f"{f'baseline (average {_score_word(view)})':<42}"
        f"{_fixed(view.baseline, places):>12}"
    )
    page.row(f"{'+ contributions':<42}{_fixed(total, places, sign=True):>12}")
    page.row(f"{f'= {view.unit}':<42}{_fixed(view.score, places):>12}")
    if view.task == "classification":
        if view.unit == "log-odds":
            page.row(f"{'→ probability':<42}{view.prediction:>12.3f}")
        page.row(f"{'→ class':<42}{view.label:>12d}")
    return page.text()
