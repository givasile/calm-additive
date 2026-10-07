# CALM: interpretable by design

Accurate locally additive models with conditional feature effects for tabular data.
Python package for the NeurIPS 2026 paper
*Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects*
([arXiv](https://arxiv.org/abs/2602.16503)).

**Project page:** https://givasile.github.io/calm-additive
**Documentation:** https://givasile.github.io/calm-additive/docs/

```bash
pip install calm-additive
```

```python
from calm_additive import CALMRegressor

model = CALMRegressor().fit(X_train, y_train)
model.predict(X_test)
model.print_summary()
```

Built on [effector](https://github.com/givasile/effector).

## Acknowledgments

The work leading to these results has been funded by the European Union under Grant Agreement No. 101289094 (project AIRIS). Views and opinions expressed are however those of the authors and do not necessarily reflect those of the European Union or the granting authority (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.

The research leading to this work has received funding from the European Union’s Horizon Europe research and innovation program under Grant Agreement No: 101135826 (aidapt.eu).
