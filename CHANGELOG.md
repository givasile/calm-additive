# Changelog

# [0.1.0] - 2026-10-06

First release on PyPI.

- `CALMRegressor` and `CALMClassifier`: scikit-learn-style estimators that fit a locally additive model — a GAM whose feature effects may depend on a few conditions on other features — to a black-box teacher (XGBoost by default), with EBM curves.
- Views on the fitted model: `print_summary`, `print_prediction`, `plot_curves`, `plot_importance`, `plot_prediction`, and the numbers behind them (`curves`, `contributions`, `importances`).
- `fit_chain` / `fit_partitions` for a selection chain or partitions built by hand with [effector](https://github.com/givasile/effector) (>= 0.6.0).
