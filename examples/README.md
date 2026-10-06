# Examples

| script | what it shows | time |
|---|---|---|
| `01_quickstart.py` | a synthetic problem with one known interaction: fit, compare with a GAM and the black box, read the ledger, plot the curves | ~15 s |
| `02_real_data.py` | the same on Bike Sharing, California Housing, Phoneme or your own CSV, with the main knobs as flags | ~30 s |

Each script has a `_repl.py` twin with the same code laid out for IPython:
no `main`, the flags as plain variables, and `# %%` blocks to send to the
REPL one at a time.

```
python examples/01_quickstart.py
python examples/02_real_data.py --dataset bike --max-depth 1
python examples/02_real_data.py --csv my.csv --target price
```

The API overview, the guides and the reference are in the documentation:
`make docs-serve` from the repo root, then http://127.0.0.1:8000.
