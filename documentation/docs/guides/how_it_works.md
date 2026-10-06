---
title: How it works
---

???+ success "Description"

    What a CALM is, the three steps `fit` runs, and how to read the numbers
    it reports.

???+ note "Reading time"

    Approx. 5' to read.

## A GAM with more than one curve per feature

A GAM predicts with a sum of one curve per feature:

\[ \hat{y}(\mathbf{x}) = \beta_0 + \sum_{i=1}^{d} f_i(x_i) \]

A CALM lets a feature have several curves and picks one by looking at the
other features:

\[ \hat{y}(\mathbf{x}) = \beta_0 + \sum_{i=1}^{d} f_i^{(r_i(\mathbf{x}_{-i}))}(x_i) \]

`r_i` is a small decision tree over the *other* features. Each leaf is a
**region**, described by threshold rules such as `workingday = True and
year = 1`, and each region has its own curve. Inside a region the model is
still additive, which is why every plot stays one-dimensional.

## Step 1: fit a teacher

`fit` first trains a black box on the data (XGBoost by default). It is used
only to find where features interact; it does not make the final predictions.

## Step 2: find the regions, keep the ones that matter

For each feature, the teacher is read with a feature-effect method (PDP by
default). When a feature interacts with others, its effect differs from row
to row; that spread is its **heterogeneity**.

1. **Propose.** For each feature with high heterogeneity, grow a partition
   tree of depth `max_partition_depth` over the other features, choosing the
   splits that reduce the heterogeneity most. A split is made only if it
   reduces it by at least `min_heterogeneity_drop`.
2. **Select.** Start from the GAM. Each round, add the one feature's tree
   that makes an additive surrogate explain the teacher best, on top of the
   trees already added. Stop when no tree adds at least `min_r2_gain`, or
   when the next tree would bring the model above
   `max_conditional_interactions`.

The selection is what keeps a CALM small: two features often describe the
same interaction, and only one of them needs the split.

## Step 3: fit the curves

With the regions fixed, one additive model is fitted on the labels, with one
curve per (feature, region). By default this is an EBM without interactions.

## Reading the numbers

| you see | it means |
|---|---|
| `R²` in `chain_.show()` | how much of the **teacher** an additive surrogate explains, on the training data |
| `ΔR²` | what one split added, on top of the splits before it |
| `solo` | what that split would add alone, on top of the GAM |
| `heter 0.76 → 0.15` | the feature's heterogeneity before and after its split |
| `n_conditional_interactions_` | number of (feature, conditioning feature) pairs |

!!! warning "The ledger certifies fidelity, not accuracy"

    A split is kept because it explains the teacher better. Whether the
    final model predicts better is measured on held-out data, as in the
    [quickstart](./quickstart.md#calm-closes-the-gap-to-the-black-box).

!!! note "A CALM cannot beat its teacher's knowledge of interactions"

    The regions come from the teacher. Where the teacher has found no
    interaction, CALM stays a GAM.

## A prediction, by hand

The baseline, plus one value per feature: the feature's curve for the
region the row falls in, read at the row's value.
[`print_prediction`](./../api_docs/api_views.md) prints that sum for
one row, [`plot_curves`](./../api_docs/api_views.md) draws the curves, and
`contributions(X).sum(axis=1) + baseline_` reproduces `predict` exactly.

---

## Where to next

- [Choosing the knobs](./choosing_the_knobs.md): the settings behind each step
- [API: parameters](./../api_docs/api_config.md): every argument, with its default
