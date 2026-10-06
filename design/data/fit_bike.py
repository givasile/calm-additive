"""Fit GAM / CALM / GA2M on Bike Sharing (full data, seed 42) and export the hour effect for the landing-page figures.

    .venv/bin/python design/data/fit_bike.py        # writes design/data/bike_{depth1,default}.json

CALM is `calm_additive.CALMRegressor` with a partition on the hour only (`partition_features=["hour"]`):
when other features are split on the hour too, their offsets show up as steps in the hour's curves.
The GAM is an EBM without interactions and the GA2M an EBM with `interactions=0.9` (0.9 x d pairs),
the setting of the paper's EB2M baseline.
"""
import json, pathlib, warnings
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.datasets import fetch_openml
from calm_additive import CALMRegressor

warnings.filterwarnings("ignore")
here = pathlib.Path(__file__).resolve().parent
df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
X, y = df.drop(columns="count"), df["count"].astype(float)
names = list(X.columns)
hours = np.arange(24)


def step_at(e, v):
    """A step curve from `curves()` (cut points repeated) read at the values `v`."""
    x, yv = np.asarray(e["x"], float), np.asarray(e["y"], float)
    return [float(yv[np.flatnonzero((x[:-1] < h) & (h < x[1:]))[0]]) for h in v]


def calm(depth):
    m = CALMRegressor(max_partition_depth=depth, partition_features=["hour"], random_state=42).fit(X, y)
    c = m.curves()
    return {
        "partitions": {f: [e["region"] for e in ents] for f, ents in c.items() if len(ents) > 1},
        "calm_hr": [{"region": e["region"], "n": int(e["n"]), "values": step_at(e, hours)} for e in c["hour"]],
        "baseline": float(m.baseline_),
    }


def grid(**fixed):
    """24 rows, one per hour; the other features as in the first row, or as given."""
    G = X.iloc[[0] * 24].copy().reset_index(drop=True)
    G["hour"] = hours
    for k, v in fixed.items():
        G[k] = np.array([v] * 24)
        G[k] = G[k].astype(X[k].dtype)
    return G


def term(model, feats):
    want = {names.index(f) for f in feats}
    hit = [i for i, t in enumerate(model.term_features_) if set(t) == want]
    return hit[0] if hit else None


out = {"hours": hours.tolist(), "unit": "rentals per hour (effect)", "names": names}

gam = ExplainableBoostingRegressor(interactions=0, random_state=42).fit(X, y)
out["gam_hr"] = gam.eval_terms(grid())[:, term(gam, ["hour"])].tolist()

g2 = ExplainableBoostingRegressor(interactions=0.9, random_state=42).fit(X, y)
out["ga2m_terms"] = [[names[k] for k in t] for t in g2.term_features_]
out["ga2m_hr_main"] = g2.eval_terms(grid())[:, term(g2, ["hour"])].tolist()
tp = term(g2, ["hour", "workingday"])
if tp is not None:
    out["ga2m_hr_x_workingday"] = {lvl: g2.eval_terms(grid(workingday=lvl))[:, tp].tolist() for lvl in ("False", "True")}

for depth, name in ((1, "bike_depth1"), (2, "bike_default")):
    d = dict(out, **calm(depth))
    json.dump(d, open(here / f"{name}.json", "w"), indent=1)
    print(name, json.dumps(d["partitions"], ensure_ascii=False))
pairs = [t for t in out["ga2m_terms"] if len(t) == 2]
print("ga2m pairs:", len(pairs), "| with hour:", [t for t in pairs if "hour" in t], "| hour x workingday:", tp is not None)
