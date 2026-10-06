"""CALM in two minutes, on a synthetic problem with a known interaction.

    python examples/01_quickstart.py

The target has one interaction: the effect of x3 is a sine when x2 >= 0 and a
cosine when x2 < 0. A GAM has to average the two; CALM should find the split
"x3 where x2 < 0 / x2 >= 0" and fit one curve per side.

The script fits a CALM, prints the model card (with where it stands between
a GAM and the black box on held-out data), why it split what it split, one
prediction taken apart, and saves the fitted curves to `quickstart.png`.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calm_additive import CALMRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# -- data: 3 features, one interaction (x2 switches the shape of x3) ---------
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.uniform(-1, 1, (3000, 3)), columns=["x1", "x2", "x3"])
y = (
    X["x1"] ** 2
    + np.log(np.abs(X["x2"]))
    + 2 * np.where(X["x2"] >= 0, np.sin(np.pi / 2 * X["x3"]), np.cos(np.pi / 2 * X["x3"]))
    + rng.normal(0, 0.3, len(X))
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -- fit ----------------------------------------------------------------------
# Defaults: an XGBoost black box as the teacher, PDP-based region detection,
# an EBM (no interactions) fitted inside the regions. `max_partition_depth`
# is the main knob: at most 2**max_partition_depth curves per feature.
calm = CALMRegressor(max_partition_depth=1)
calm.fit(X_train, y_train)

# -- the model on one screen ---------------------------------------------------
# What it is made of, which features matter, each partition as a tree — and,
# because held-out data is passed, where it stands between a GAM (the same
# curves, no partitions) and the black box it learned from.
calm.print_summary(X_test, y_test)

# -- why it has this structure -------------------------------------------------
# The ledger: R2 here is how much of the *black box* the additive surrogate
# explains. Each accepted partition must add at least `min_r2_gain` (default 1%).
calm.chain_.show()

# -- one prediction, taken apart -----------------------------------------------
# baseline + one contribution per feature = the prediction, exactly; x3's line
# names the region (the side of x2) whose curve was read.
calm.print_prediction(X_test.iloc[0])

# -- the model itself: one panel per feature, one curve per region ------------
fig, _ = calm.plot_curves(show_plot=False)
fig.savefig("quickstart.png", dpi=130)
fig, _ = calm.plot_curves(at=X_test.iloc[0], show_plot=False)
fig.savefig("quickstart_row.png", dpi=130)
print("\nfitted curves saved to quickstart.png, quickstart_row.png")
