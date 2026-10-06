# API Docs

- [Estimators](./api_docs/api_estimators.md)
    - [`CALMRegressor`](./api_docs/api_estimators.md#calm_additive.engine.CALMRegressor) — CALM for regression
    - [`CALMClassifier`](./api_docs/api_estimators.md#calm_additive.engine.CALMClassifier) — CALM for binary classification
- [Parameters](./api_docs/api_config.md) — every argument, by importance and by step
- [Printing and plotting](./api_docs/api_views.md)
    - [`print_summary`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.print_summary) — the model card; with data, where it stands between a GAM and the black box
    - [`plot_curves`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.plot_curves) — one panel per feature, one curve per region
    - [`plot_importance`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.plot_importance) — the features by importance
    - [`print_prediction`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.print_prediction), [`plot_prediction`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.plot_prediction) — one prediction, taken apart
    - [`curves`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.curves), [`contributions`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.contributions), [`importances`](./api_docs/api_views.md#calm_additive.engine.CALMRegressor.importances) — the numbers behind them
- [Extending](./api_docs/api_extending.md)
    - [`register_teacher`](./api_docs/api_extending.md#calm_additive.engine.register_teacher) — add a teacher
    - [`register_curve_fitter`](./api_docs/api_extending.md#calm_additive.engine.register_curve_fitter) — add a regional model
    - [`fit_chain`](./api_docs/api_extending.md#calm_additive.engine.fit_chain), [`fit_partitions`](./api_docs/api_extending.md#calm_additive.engine.fit_partitions) — the lower-level entry points
