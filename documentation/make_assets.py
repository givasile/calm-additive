"""Regenerate what the documentation pages show.

    python documentation/make_assets.py

Runs the code of the two example scripts once. Figures go to
`documentation/docs/static/`; printed output goes to
`documentation/_outputs/*.txt` (git-ignored) to be pasted into the pages, so
no page shows a number that was not printed by the package.
"""

import contextlib
import io
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
STATIC = HERE / "docs" / "static"
OUT = HERE / "_outputs"

# name -> (script, arguments, prefix of the figures the script saves)
RUNS = {
    "quickstart": ("01_quickstart.py", [], "quickstart"),
    "bike": ("02_real_data.py", ["--dataset", "bike"], "bike_"),
    "phoneme": ("02_real_data.py", ["--dataset", "phoneme"], "phoneme_"),
}


def main():
    import os

    STATIC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    os.chdir(OUT)  # the scripts save their figures in the working directory
    for name, (script, argv, prefix) in RUNS.items():
        buf = io.StringIO()
        sys.argv = [script] + argv
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(EXAMPLES / script), run_name="__main__")
        (OUT / f"{name}.txt").write_text(buf.getvalue())
        figures = sorted(OUT.glob(f"{prefix}*.png"))
        for figure in figures:
            figure.replace(STATIC / figure.name)
        print(f"{name}: {OUT / (name + '.txt')}  +  {[f.name for f in figures]}")


if __name__ == "__main__":
    main()
