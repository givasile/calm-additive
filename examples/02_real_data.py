"""CALM on a real dataset, with the main knobs on the command line.

    python examples/02_real_data.py                           # Bike Sharing
    python examples/02_real_data.py --dataset california
    python examples/02_real_data.py --dataset phoneme         # classification
    python examples/02_real_data.py --csv my.csv --target price
    python examples/02_real_data.py --max-partition-depth 1 --min-r2-gain 0.02
    python examples/02_real_data.py --curve-fitter surrogate  # no refit, fast

Prints the model card — with the GAM / CALM / black-box comparison on a
held-out split — the ledger of accepted and rejected splits, one prediction
taken apart, and saves the figures as `<name>_*.png`. The built-in datasets
are downloaded by scikit-learn on first use.
"""

import argparse
import time
import warnings

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

from calm_additive import CALMClassifier, CALMRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def load(args):
    """-> (name, X DataFrame, y Series, task); the Series' name is the
    target's name in everything CALM prints and draws."""
    if args.csv:
        df = pd.read_csv(args.csv).dropna()
        y = df.pop(args.target)
        task = args.task or ("classification" if y.nunique() == 2 else "regression")
        if task == "classification":
            y = pd.Series(pd.factorize(y, sort=True)[0], index=y.index, name=y.name)
        return args.target, df, y, task
    from sklearn.datasets import fetch_california_housing, fetch_openml

    if args.dataset == "bike":
        df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
        return "bike", df.drop(columns="count"), df["count"].astype(float), "regression"
    if args.dataset == "california":
        d = fetch_california_housing(as_frame=True)
        return "california", d.data, d.target, "regression"
    df = fetch_openml("phoneme", version=1, as_frame=True).frame
    y = (df.pop("Class") == "2").astype(int).rename("oral")  # 1 = nasal, 2 = oral
    return "phoneme", df, y, "classification"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["bike", "california", "phoneme"], default="bike")
    ap.add_argument("--csv", help="your own data: a CSV file (use with --target)")
    ap.add_argument("--target", help="target column of --csv")
    ap.add_argument("--task", choices=["regression", "classification"],
                    help="for --csv; default: classification if the target has 2 values")
    # the knobs
    ap.add_argument("--max-partition-depth", type=int, default=2,
                    help="depth of each feature's partition tree: at most 2**depth curves per feature")
    ap.add_argument("--min-r2-gain", type=float, default=0.01,
                    help="a partition is kept only if it explains this much more of the teacher")
    ap.add_argument("--curve-fitter", choices=["ebm", "surrogate"], default="ebm",
                    help="ebm: refit an EBM inside the regions (the paper); "
                         "surrogate: reuse the teacher's regional curves, no refit")
    ap.add_argument("--selection-target", choices=["teacher", "y"], default="teacher",
                    help="score candidate partitions against the teacher or against y")
    ap.add_argument("--partition-features", choices=["heterogeneous", "all"], default="heterogeneous",
                    help="which features may get a partition: above-median heterogeneity, or all")
    args = ap.parse_args()
    if args.csv and not args.target:
        ap.error("--csv needs --target")

    name, X, y, task = load(args)
    clf = task == "classification"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y if clf else None
    )
    print(f"\n{name}: {len(X)} rows, {X.shape[1]} features, {task}")

    Calm = CALMClassifier if clf else CALMRegressor

    t = time.time()
    calm = Calm(
        max_partition_depth=args.max_partition_depth,
        min_r2_gain=args.min_r2_gain,
        partition_features=args.partition_features,
        curve_fitter=args.curve_fitter,
        selection_target=args.selection_target,
    )
    calm.fit(X_train, y_train)
    t_calm = time.time() - t

    print(f"CALM fitted in {t_calm:.1f} s")

    # the model on one screen; with held-out data, also where it stands
    # between a GAM (same curve fitter, no partitions) and the black box
    calm.print_summary(X_test, y_test)

    # why it has this structure: accepted and rejected partitions
    if calm.chain_ is not None:
        calm.chain_.show()

    # one prediction, taken apart: the held-out row with the highest score
    row = X_test.iloc[int(calm.predict_score(X_test).argmax())]
    calm.print_prediction(row)

    figures = {
        "calm": calm.plot_curves(max_features=9, show_plot=False),
        "importance": calm.plot_importance(show_plot=False),
        "prediction": calm.plot_prediction(row, show_plot=False),
        "row": calm.plot_curves(at=row, max_features=6, show_plot=False),
    }
    if calm.partitions_:  # the most important feature with a partition, large
        names = list(calm.feature_names_in_)
        top = max(calm.partitions_, key=lambda j: calm.importances()[j])
        figures["feature"] = calm.plot_curves(names[top], show_plot=False)
    for key, (fig, _) in figures.items():
        fig.savefig(f"{name}_{key}.png", dpi=130)
    print(f"\nfigures saved to {name}_*.png: {', '.join(figures)}")


if __name__ == "__main__":
    main()
