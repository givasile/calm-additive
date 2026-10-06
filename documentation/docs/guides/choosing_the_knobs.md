---
title: Choosing the knobs
---

???+ success "Description"

    Which setting to change for which situation. Every setting is a keyword
    argument of `CALMRegressor(...)` / `CALMClassifier(...)`; the
    [parameters page](./../api_docs/api_config.md) lists them all.

???+ note "Reading time"

    Approx. 3' to read.

## Start from the defaults

```python
from calm_additive import CALMRegressor

calm = CALMRegressor(
    max_conditional_interactions=10,
    max_partition_depth=2,
    min_heterogeneity_drop=0.05,
    min_r2_gain=0.01,
)
```

## The four that matter most

### `max_conditional_interactions`

The size limit (K in the paper): the most conditional interactions the model
may have. A conditional interaction is one (feature, conditioning feature)
pair, so `x1 | x2` and `x2 | x1` are two. A feature's partition is kept
whole or not at all, so the selection stops before the partition that would
exceed the limit. `None` removes the limit.

### `max_partition_depth`

The accuracy against simplicity dial (d_max in the paper). It is the depth
of each feature's partition tree: each level doubles the curves a feature
may have, and roughly doubles the conditional interactions.

### `min_heterogeneity_drop`

The bar a split must clear inside one partition tree (ε in the paper): the
fraction by which it must reduce the feature's heterogeneity. Raise it and
the trees stop splitting earlier.

### `min_r2_gain`

The bar a whole partition must clear to be kept. It is a fraction of the
teacher's variance, so `0.01` means one percentage point of R².

👉 Change one knob at a time and read `calm.chain_.show()` after each fit:
the ledger shows which partitions appeared or disappeared, and what each was
worth.

## Your case, what to set

| your case | use |
|---|---|
| the plots have too many curves | `max_partition_depth=1`: at most two curves per feature |
| CALM is far from the black box and you can afford more curves | `max_partition_depth=3` |
| too many features are split | raise `min_r2_gain`, e.g. `0.02`, or set `max_partitions` |
| you want at most a given number of conditional interactions | `max_conditional_interactions=5` |
| you want no limit on the size, as in the paper | `max_conditional_interactions=None` |
| a split you expected is missing | lower `min_r2_gain`, or `partition_features="all"` |
| you want the regions judged on the labels, not on the teacher | `selection_target="y"` |
| you want every partition the trees find, without selection | `selection="all"`, with `max_conditional_interactions=None` |
| rules should use numeric features only | `conditioning_features="numeric"` |
| regions are too small to trust | `partitioner_params={"min_samples_leaf": 50}` |
| features are strongly correlated | `effect="ale"` |
| you already have a trained model | `teacher=my_model`, or `teacher=my_model.predict` |
| you want a stronger or lighter teacher | `teacher_params={"n_estimators": 1000}` |
| the EBM step is slow | `curve_fitter_params={"learning_rate": 0.1, "outer_bags": 4}`: on Bike 4 s instead of 14 s, same test error, slightly rougher curves |
| you want a result fast, without refitting | `curve_fitter="surrogate"` |

## What the curve fitter changes

| `curve_fitter` | what happens | when |
|---|---|---|
| `"ebm"` (default) | an EBM without interactions is refitted on the labels, inside the regions | the model you report |
| `"surrogate"` | the teacher's own regional curves are frozen into a model, no refit | a fast look at the structure |

On Bike Sharing the refit matters: test RMSE 61 with `"ebm"`, 84 with
`"surrogate"` (selection on labels, all features splittable).

---

## Where to next

- [API: parameters](./../api_docs/api_config.md): every argument, with its default
- [How it works](./how_it_works.md): the steps the knobs control
