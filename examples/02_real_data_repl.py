# CALM on a real dataset — REPL version of 02_real_data.py.
#
# Same code, laid out for IPython: the command-line flags are plain variables
# in the "settings" block, and every block between two `# %%` lines is
# top-level code, so mark it and send it to the REPL. Change a setting, then
# re-send the blocks from "fit CALM" down.

# %% imports
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

import calm_additive
from calm_additive import CALMClassifier, CALMRegressor

warnings.filterwarnings("ignore")

# %% settings
DATASET = "california"              # "bike" | "california" | "phoneme" | "csv"
CSV, TARGET = None, None      # for DATASET = "csv": file path and target column

MAX_PARTITION_DEPTH = 2       # at most 2**depth curves per feature
MIN_R2_GAIN = 0.01            # a partition must explain this much more of the teacher
CURVE_FITTER = "ebm"          # "ebm": refit inside the regions | "surrogate": no refit
SELECTION_TARGET = "teacher"  # score partitions against the "teacher" or "y"
PARTITION_FEATURES = "heterogeneous"  # which features may get a partition: "heterogeneous" | "all"

# %% load the data  ->  name, X (DataFrame), y (array), task
if DATASET == "bike":
    df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    name, X, y, task = "bike", df.drop(columns="count"), df["count"].to_numpy(float), "regression"
elif DATASET == "california":
    d = fetch_california_housing(as_frame=True)
    name, X, y, task = "california", d.data, d.target.to_numpy(), "regression"
elif DATASET == "phoneme":
    df = fetch_openml("phoneme", version=1, as_frame=True).frame
    y = (df.pop("Class") == "2").to_numpy(int)
    name, X, task = "phoneme", df, "classification"
else:
    df = pd.read_csv(CSV).dropna()
    y = df.pop(TARGET)
    task = "classification" if y.nunique() == 2 else "regression"
    y = pd.factorize(y, sort=True)[0] if task == "classification" else np.asarray(y)
    name, X = TARGET, df

clf = task == "classification"
print(f"{name}: {len(X)} rows, {X.shape[1]} features, {task}")

# %% split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y if clf else None
)

# %% fit CALM
Calm = CALMClassifier if clf else CALMRegressor
t = time.time()
calm = Calm(
    effect="ale",
    max_partitions=1,
    max_partition_depth=2,
    min_r2_gain=0,
    partition_features="all",
    curve_fitter="ebm",
    selection_target="teacher",
    verbose=1,
)
calm.fit(X_train, y_train)
t_calm = time.time() - t

# %% fit the GAM reference
# CALM encodes the DataFrame's categorical columns itself; the two reference
# models get the same numeric matrix
Xn_train, Xn_test = calm._encode(X_train), calm._encode(X_test)
Gam = ExplainableBoostingClassifier if clf else ExplainableBoostingRegressor
t = time.time()
gam = Gam(interactions=0, random_state=42).fit(Xn_train, y_train)
t_gam = time.time() - t

# %% accuracy on the held-out split
preds = {
    "GAM (EBM, no interactions)": (gam.predict(Xn_test), t_gam),
    "CALM": (calm.predict(X_test), t_calm),
    "black box (XGBoost teacher)": (calm.teacher_.predict(Xn_test), None),
}
metric = "accuracy" if clf else "RMSE      R2"
print(f"  {'model':30s}{metric:>14s}   fit time")
for label, (p, secs) in preds.items():
    score = (
        f"{accuracy_score(y_test, p):14.3f}" if clf
        else f"{root_mean_squared_error(y_test, p):9.3f}{r2_score(y_test, p):8.3f}"
    )
    took = f"{secs:6.1f} s" if secs is not None else "  (inside CALM)"
    print(f"  {label:30s}{score:>14s}   {took}")

# %% the ledger: accepted and rejected splits
calm.chain_.show()

# %% the regions per feature
names = calm.effect_.feature_names
print(f"conditional interactions: {calm.n_conditional_interactions_}")
for j, part in calm.partitions_.items():
    print(f"  {names[j]}: {len(part.leaves)} regions")
    for leaf in part.leaves:
        print(f"      {part.label(leaf.idx)}")

# %% the fitted curves: split features first
split = [names[j] for j in calm.partitions_]
rest = [n for n in names if n not in split]
fig, axes = calm.plot_curves(features=(split + rest)[:9], show_plot=False)
plt.show()

# %% only the split features, larger
fig, axes = calm.plot_curves(features=split, ncols=2, show_plot=False)
plt.show()

# # %% the new views: the card, one prediction, importance
calm.print_summary(X_test, y_test)             # the card + GAM / CALM / black box
calm.print_prediction(X_test.iloc[21])          # one row's ledger
calm.plot_prediction(X_test.iloc[21])           # … as a waterfall
calm.plot_curves(at=X_test.iloc[21], max_features=4, ncols=2)            # … on the curves
calm.plot_importance()

# # %% things to poke at
# curves = calm.curves()                         # {feature: [{"region", "x", "y", "n"}, ...]}
# calm.interactions_info_                        # regions per feature, conditioning features
calm.chain_.final.show()                       # the final snapshot, with each feature's tree
calm.effect_.plot(split[0])                    # the teacher's view of the first split feature
