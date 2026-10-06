"""Accuracy of GAM / CALM / GA2M / black box on the landing page's Illustration.

    .venv/bin/python design/data/fit_synth.py        # writes design/data/synth_accuracy.json

The function of the hand-drawn trio: the effect of x3 is `x3` when x2 > 0 and `0.3 - 0.9 x3^2` when
x2 <= 0; x1 adds a main effect; noise 0.1. 5000 rows, 5-fold cross-validation, RMSE on the held-out fold.
"""
import json, pathlib, warnings
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from calm_additive import CALMRegressor

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
X = pd.DataFrame(rng.uniform(-1, 1, (5000, 3)), columns=["x1", "x2", "x3"])
y = X["x1"] + np.where(X["x2"] > 0, X["x3"], 0.3 - 0.9 * X["x3"] ** 2) + rng.normal(0, 0.1, len(X))
models = {
    "gam": lambda: ExplainableBoostingRegressor(interactions=0, random_state=42),
    "calm": lambda: CALMRegressor(random_state=42),
    "ga2m": lambda: ExplainableBoostingRegressor(interactions=0.9, random_state=42),
    "blackbox": lambda: XGBRegressor(random_state=42),
}
rmse = {k: [] for k in models}
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    for k, make in models.items():
        m = make().fit(X.iloc[tr], y.iloc[tr])
        rmse[k].append(float(np.sqrt(np.mean((m.predict(X.iloc[te]) - y.iloc[te]) ** 2))))
out = {"metric": "RMSE", "noise": 0.1, "rmse": {k: float(np.mean(v)) for k, v in rmse.items()}, "std": {k: float(np.std(v)) for k, v in rmse.items()}}
json.dump(out, open(pathlib.Path(__file__).resolve().parent / "synth_accuracy.json", "w"), indent=1)
print(json.dumps(out["rmse"]), json.dumps(out["std"]))
