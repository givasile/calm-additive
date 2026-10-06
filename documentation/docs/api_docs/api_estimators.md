---
title: Estimators
---

## Summary

`CALMRegressor` and `CALMClassifier` are the two estimators. One `fit` call
trains the teacher, finds and selects the regions, and fits one curve per
(feature, region). Both follow the scikit-learn convention: constructor
arguments configure, `fit` learns, fitted attributes end with `_`. Every
argument is keyword-only and has a default; the
[parameters page](./api_config.md) lists them by importance.

---

## Usage

```python
from calm_additive import CALMRegressor, CALMClassifier

calm = CALMRegressor(max_partition_depth=2)
calm.fit(X_train, y_train)      # numpy array or pandas DataFrame
calm.predict(X_test)

clf = CALMClassifier().fit(X_train, y_train)   # y in {0, 1}
clf.predict(X_test)             # labels
clf.predict_proba(X_test)       # (N, 2)
```

After `fit`:

| access | what it is |
|---|---|
| `calm.print_summary()`, `calm.plot_curves()`, … | the model printed and drawn: see [printing and plotting](./api_views.md) |
| `calm.chain_.show()` | the ledger: accepted and rejected partitions, and why |
| `calm.n_conditional_interactions_` | number of (feature, conditioning feature) pairs; `x1 \| x2` and `x2 \| x1` are two |
| `calm.partitions_` | `{feature index: Partition}`; `part.leaves`, `part.label(leaf.idx)` |
| `calm.interactions_info_` | regions per feature, conditioning features |
| `calm.teacher_` | the fitted teacher |
| `calm.effect_` | the effector effect used for detection |
| `calm.model_` | the fitted additive model |
| `calm.gam_` | the GAM reference, once `print_summary(X, y)` has fitted it |

`chain_`, `partitions_` and `effect_` are [effector](https://xai-effector.github.io)
objects and are documented there.

!!! note "Binary classification and regression only"

    Multi-class targets are not supported.

## API

### ::: calm_additive.engine.CALMRegressor
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - fit
          - predict
          - predict_score
          - paper

### ::: calm_additive.engine.CALMClassifier
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - fit
          - predict
          - predict_proba
          - predict_score
          - paper
