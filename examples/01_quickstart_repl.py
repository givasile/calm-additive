# CALM in two minutes — REPL version of 01_quickstart.py.
#
# Same code, laid out for IPython: every block between two `# %%` lines is
# independent top-level code, so mark it and send it to the REPL.
#
# The target has one interaction: the effect of x3 is a sine when x2 >= 0 and
# a cosine when x2 < 0. A GAM has to average the two; CALM should find the
# split "x3 where x2 < 0 / x2 >= 0" and fit one curve per side.

# %% imports
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calm_additive import CALMRegressor

warnings.filterwarnings("ignore")

# %% data: 3 features, one interaction (x2 switches the shape of x3)
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.uniform(-1, 1, (3000, 3)), columns=["x1", "x2", "x3"])
y = (
    X["x1"] ** 2
    + np.log(np.abs(X["x2"]))
    + 2 * np.where(X["x2"] >= 0, np.sin(np.pi / 2 * X["x3"]), np.cos(np.pi / 2 * X["x3"]))
    + rng.normal(0, 0.3, len(X))
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% fit CALM
# Defaults: an XGBoost black box as the teacher, PDP-based region detection,
# an EBM (no interactions) fitted inside the regions. `max_partition_depth`
# is the main knob: at most 2**max_partition_depth curves per feature.
calm = CALMRegressor(max_partition_depth=1)
calm.fit(X_train, y_train)

# %% the model on one screen
# What it is made of, which features matter, each partition as a tree — and,
# because held-out data is passed, where it stands between a GAM (the same
# curves, no partitions) and the black box it learned from.
calm.print_summary(X_test, y_test)

# %% why it has this structure
# The ledger: R2 here is how much of the *black box* the additive surrogate
# explains. Each accepted partition must add at least `min_r2_gain` (default 1%).
calm.chain_.show()

# %% the model itself: one panel per feature, one curve per region
calm.plot_curves()

# %% one feature, large
calm.plot_curves("x3")

# %% which features carry the model
calm.plot_importance()

# %% one prediction, taken apart: printed, as a waterfall, and on the curves
row = X_test.iloc[0]
calm.print_prediction(row)
calm.plot_prediction(row)
calm.plot_curves(at=row)

# %% the numbers behind the views
curves = calm.curves()                  # {feature: [{"region", "x", "y", "n"}, ...]}
contrib = calm.contributions(X_test)    # (N, D); rows sum to predict - baseline_
calm.importances()                      # (D,) mean |contribution|
calm.baseline_                          # the average prediction

# %% things to poke at
calm.partitions_            # {feature index: Partition}
calm.interactions_info_     # regions per feature, conditioning features
calm.chain_.final.show()    # the final snapshot, with each feature's tree
calm.effect_.plot("x3")     # the teacher's view of x3 (PDP + ICE)
