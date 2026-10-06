---
title: Printing and plotting
---

## Summary

A fitted CALM is its own explanation: `score = baseline + one curve value per
feature`, the curve being the one of the region the row falls in. Five
methods show it. The ones named `print_…` print, the ones named `plot_…`
draw.

| method | answers |
|---|---|
| [`print_summary()`](#calm_additive.engine.CALMRegressor.print_summary) | what is this model: size, features by importance, partitions; with data, where it stands between a GAM and the black box |
| [`plot_curves()`](#calm_additive.engine.CALMRegressor.plot_curves) | what does each feature do, and where: one panel per feature, one curve per region |
| [`plot_importance()`](#calm_additive.engine.CALMRegressor.plot_importance) | which features carry the model |
| [`print_prediction(x)`](#calm_additive.engine.CALMRegressor.print_prediction) | why this number, for one row |
| [`plot_prediction(x)`](#calm_additive.engine.CALMRegressor.plot_prediction) | the same row as a waterfall |

Three more return the numbers behind them:
[`curves()`](#calm_additive.engine.CALMRegressor.curves),
[`contributions(X)`](#calm_additive.engine.CALMRegressor.contributions) and
[`importances()`](#calm_additive.engine.CALMRegressor.importances).

`CALMClassifier` has the same methods.

---

## Usage

```python
calm.print_summary()                  # the model card
calm.print_summary(X_test, y_test)    # … with GAM / CALM / black box on that data

calm.plot_curves()                    # the most important features
calm.plot_curves("hour")              # one feature, large
calm.plot_importance()

row = X_test.iloc[0]
calm.print_prediction(row)            # the row's ledger
calm.plot_prediction(row)             # … as a waterfall
calm.plot_curves(at=row)              # … marked on the curves

fig, axes = calm.plot_curves(show_plot=False)   # to save or edit
```

## What the views say

**Everything is read from the fitted model.** Curves, contributions and
importances come from the curves `fit` produced, not from the teacher, and
they are exact: for every row,

```python
calm.contributions(X).sum(axis=1) + calm.baseline_   # == the model's score
```

The score is the prediction for regression. For a classifier it is the
log-odds of the positive class, the scale the model is additive on; the
ledger's last lines map it to the probability and the class.

**A curve is a feature's whole contribution inside its region**, centred so
that the feature's contributions average zero over the training rows. The
baseline is therefore the average score. Curves of the same feature sit at
different heights when their regions differ in level.

**Curves are drawn where the data is.** Each curve covers only the values
its region has training rows for, and the strip under a panel shows how many
(one row of bars per region). Categorical features are drawn as one mark per
level, with the level names on the axis.

**Importance** is the mean |contribution| of a feature over the training
rows, in the units of the score. The bar of a feature with a partition is
cut into one segment per region.

**Where it stands.** `print_summary(X, y)` scores three models on the data you
pass: a GAM (the same curve fitter, with no partitions), this CALM, and the
teacher. The GAM is fitted on the training data the first time it is needed
and kept as `calm.gam_`. Pass held-out data: on its own training rows the
teacher looks better than it is.

!!! note "Why the model has this structure"

    These views describe the model. Why a partition was kept or rejected is
    the selection ledger, `calm.chain_.show()`, and what the teacher showed
    for a feature is `calm.effect_.plot("hour")`; both are
    [effector](https://xai-effector.github.io) objects.

!!! note "The look"

    Figures use effector's active theme: `effector.set_theme("dark")` or
    `"paper"` changes their colours. A schema's `scale_x_list` / `scale_y`
    is applied to the views (not to `contributions`, `importances` and
    `baseline_`, which stay in the model's units).

## API

### ::: calm_additive.engine.CALMRegressor
      options:
        show_root_heading: False
        show_root_toc_entry: False
        show_symbol_type_toc: True
        inherited_members: True
        heading_level: 3
        members:
          - print_summary
          - plot_curves
          - plot_importance
          - print_prediction
          - plot_prediction
          - curves
          - contributions
          - importances
          - baseline_
