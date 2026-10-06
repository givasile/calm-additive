---
title: Parameters
---

## Summary

Every setting is a keyword argument of `CALMRegressor` / `CALMClassifier`,
and every one has a default. They come in three tiers: the hyperparameters
to tune, the choices each step uses, and the advanced ones you can ignore.

---

## Usage

```python
from calm_additive import CALMRegressor

calm = CALMRegressor()                                  # the defaults
calm = CALMRegressor(max_partition_depth=1,             # two curves per feature
                     min_r2_gain=0.02)                  # a stricter bar
calm = CALMRegressor(teacher=my_model,                  # your own black box
                     curve_fitter_params={"max_bins": 64})
```

### 1. Hyperparameters: the ones to tune

| argument | default | paper | what it decides |
|---|---|---|---|
| `max_conditional_interactions` | 10 | K | the most conditional interactions in the model; `None` for no limit |
| `max_partitions` | `None` | | the most features that may get a partition |
| `max_partition_depth` | 2 | d_max | curves per feature: at most `2**max_partition_depth` |
| `min_heterogeneity_drop` | 0.05 | ε | how much a split must reduce a feature's heterogeneity to be made |
| `min_r2_gain` | 0.01 | | how much a feature's partition must explain to be kept |

A conditional interaction is one (feature, conditioning feature) pair:
`x1 | x2` and `x2 | x1` are two. A feature's partition is kept whole or not
at all, so the selection stops before the partition that would exceed a
limit; `calm.chain_.show()` lists what the limit left out.

📄 In depth: [choosing the knobs](./../guides/choosing_the_knobs.md).

### 2. Choices: what each step uses

| argument | default | step | what it decides |
|---|---|---|---|
| `teacher` | `"xgb"` | 1 | the black box the regions are learned from |
| `effect` | `"pdp"` | 2 | how the teacher is read: `"pdp"`, `"ale"`, `"shap"`, `"rhale"` |
| `partitioner` | `"best"` | 2 | what proposes a partition tree per feature |
| `partition_features` | `"heterogeneous"` | 2 | which features may get a partition |
| `conditioning_features` | `"all"` | 2 | which features may appear in the region rules |
| `selection` | `"r2_gain"` | 3 | keep the partitions that earn an R² gain, or `"all"` of them |
| `curve_fitter` | `"ebm"` | 4 | what is fitted inside the regions |

The steps: 1 fit the teacher, 2 propose one partition tree per feature,
3 select which trees are kept, 4 fit one curve per (feature, region).

`teacher`, `effect`, `partitioner` and `curve_fitter` are *slots*: each takes
a registered name or an object of your own.

### 3. Advanced

| argument | default | what it decides |
|---|---|---|
| `selection_target` | `"teacher"` | judge partitions against the teacher or the labels (`"y"`) |
| `min_relative_r2_gain` | `None` | a relaxed bar for nearly saturated models |
| `teacher_params` | `None` | settings of a teacher given by name, e.g. `{"n_estimators": 1000}` |
| `effect_params` | `None` | effector's settings for the effect, e.g. `{"nof_instances": 10_000}` |
| `partitioner_params` | `None` | effector's settings for the partitioner, e.g. `{"min_samples_leaf": 50}` |
| `curve_fitter_params` | `None` | settings of a fitter given by name, e.g. `{"max_bins": 64}` |
| `schema` | `None` | feature names and types for a numpy `X` |
| `random_state` | 42 | one seed for every component built by name |
| `verbose` | 0 | 1 shows a progress bar over the stages of `fit`, with the time each took |

!!! note "A `*_params` dict tunes a component given by name"

    Passing a `*_params` dict together with an object in the same slot
    raises: the object is already configured. The same holds for
    `max_partition_depth` and `min_heterogeneity_drop` with a partitioner
    object.

!!! note "The limits need a selection"

    `selection="all"` keeps every proposed partition, so it raises unless
    `max_conditional_interactions` and `max_partitions` are `None`.

The full description of each argument is on the
[estimators page](./api_estimators.md).

## The paper's preset

`CALMRegressor.paper()` / `CALMClassifier.paper()` return an estimator with
the settings of the submitted paper's method on this engine: every proposed
partition is applied, without selection and without a limit on the
conditional interactions.
