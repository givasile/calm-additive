# Examples

???+ success "Description"

    Four runnable scripts in
    [`examples/`](https://github.com/givasile/calm-additive/tree/main/examples).
    Each one comes in two layouts: a command-line script, and a REPL twin
    with the same code in `# %%` blocks to send to IPython one at a time.

## The scripts

| # | script | what it shows | time |
|---|---|---|---|
| 1 | `01_quickstart.py` | a synthetic problem with one known interaction: fit, print the model card (with the GAM and the black box), read the ledger, take one prediction apart, plot the curves | ~15 s |
| 1r | `01_quickstart_repl.py` | the same, block by block | |
| 2 | `02_real_data.py` | Bike Sharing, California Housing, Phoneme or your own CSV, with the main knobs as flags; saves the curves, the importance bars and one prediction as figures | ~40 s |
| 2r | `02_real_data_repl.py` | the same, with the flags as plain variables | |

## Running them

```bash
python examples/01_quickstart.py

python examples/02_real_data.py                           # Bike Sharing
python examples/02_real_data.py --dataset california
python examples/02_real_data.py --dataset phoneme         # classification
python examples/02_real_data.py --csv my.csv --target price
python examples/02_real_data.py --max-partition-depth 1 --min-r2-gain 0.02
python examples/02_real_data.py --curve-fitter surrogate  # no refit, fast
```

The built-in datasets are downloaded by scikit-learn on first use.

## What they print

| dataset | GAM | CALM | black box | conditional interactions |
|---|---|---|---|---|
| synthetic (R²) | 0.717 | 0.950 | 0.939 | 1 |
| Bike Sharing (RMSE) | 100.6 | 60.6 | 41.0 | 5 |
| Phoneme (accuracy) | 0.836 | 0.849 | 0.907 | 4 |

!!! note "One split, not a benchmark"

    These are single 80/20 splits on raw data, there to show the scripts
    work. They are not the paper's cross-validated results.

📄 In depth: [the quickstart guide](./guides/quickstart.md) retells script 1,
[a real dataset](./guides/real_data.md) retells script 2.
