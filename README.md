# CALM: interpretable by design

Accurate locally additive models with conditional feature effects for tabular data.
Python package for the NeurIPS 2026 paper
*Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects*
([arXiv](https://arxiv.org/abs/2602.16503)).

**Project page:** https://givasile.github.io/calm-additive

The package is being released in October 2026. Until then this repository holds the project page only.

```python
from calm_additive import CALMRegressor

model = CALMRegressor().fit(X_train, y_train)
model.predict(X_test)
model.explain()
```

Built on [effector](https://github.com/givasile/effector).
