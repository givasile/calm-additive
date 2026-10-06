---
title: Extending
---

## Summary

The estimators resolve their teacher and their regional model by name from
two registries. Registering a new name makes it available as `teacher=` or
`curve_fitter=`. Below the estimators sit two functions that turn regions you
already have into a model.

---

## Usage

A new teacher: `factory(task, **params)` returns an object with `fit(X, y)`
and `forward(X)` (values for regression, positive-class probability for
classification). `params` is the user's `teacher_params`; `random_state` is
added if the factory accepts it.

```python
from calm_additive import CALMRegressor, register_teacher

register_teacher("my_teacher", lambda task, **params: MyTeacher(**params))
calm = CALMRegressor(teacher="my_teacher", teacher_params={"depth": 6}).fit(X, y)
```

A new regional model: implement `MaskedFitter`. `X` is the masked design,
one column per (feature, region); `mask` marks which entries are active.

```python
from calm_additive import MaskedFitter, register_curve_fitter

class MyFitter(MaskedFitter):
    def fit(self, X, y, mask): ...
    def predict(self, X, mask): ...
    def curves(self): ...          # optional: what the model learned

register_curve_fitter("mine", lambda ctx, **params: MyFitter(**params))
calm = CALMRegressor(curve_fitter="mine").fit(X, y)
```

`curves()` returns the fitted model as an `AdditiveTerms`: the intercept, one
`Curve` per masked column, and what each column adds when it is inactive.
With it, [every view](./api_views.md) (`print_summary`, `plot_curves`, …) works for
your fitter; without it the model still predicts.

```python
from calm_additive import AdditiveTerms, Curve

def curves(self):
    columns = [Curve(knots, values, "linear") for knots, values in self.splines]
    return AdditiveTerms(self.intercept, columns, self.inactive_scores)
```

👉 A teacher needs no registration if you pass it directly:
`CALMRegressor(teacher=my_model)`.

## API

### ::: calm_additive.engine.register_teacher
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.register_curve_fitter
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.MaskedFitter
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.Curve
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.AdditiveTerms
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.fit_chain
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.fit_partitions
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: calm_additive.engine.ExecutableCALM
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        members:
          - predict
          - predict_score
          - terms
          - interactions_info
          - n_conditional_interactions
