"""Read the paper's result tables into design/data/paper_results.json, for the landing page's Accuracy section.

    python3 design/data/paper_tables.py [~/github/papers/CALM/latest]

Source: `generated/tab_{clf,regr}_rows.tex` (accuracy / RMSE, mean over 5 folds) and
`generated/tab_{clf,regr}_inter_rows.tex` (number of pairwise interactions), as written by the
paper's `scripts/make_tables.py`. Rerun after the tables change.
"""
import json, pathlib, re, sys

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/github/papers/CALM/latest").expanduser() / "generated"
MODELS = ["blackbox", "nam", "gam", "calm", "eb2m", "node", "gami"]  # XGB, NAM, EBM, CALM, EB2M, NODE-GA2M, GAMI-Net
INTER = ["calm", "eb2m", "node", "gami"]
num = re.compile(r"\$\s*([0-9.]+)(?:\\!\\times\\!10\^\{(-?\d+)\})?")


def rows(name, keys):
    out = {}
    for line in (src / name).read_text().splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != len(keys) + 1 or cells[0].startswith("\\textbf"):  # the W/D/L and Avg. rows
            continue
        vals = [num.search(c) for c in cells[1:]]
        out[cells[0]] = {k: float(m.group(1)) * 10 ** int(m.group(2) or 0) for k, m in zip(keys, vals)}
    return out


out = {}
for task, stem, metric in (("classification", "tab_clf", "accuracy"), ("regression", "tab_regr", "rmse")):
    score, inter = rows(f"{stem}_rows.tex", MODELS), rows(f"{stem}_inter_rows.tex", INTER)
    assert set(score) == set(inter), set(score) ^ set(inter)
    out[task] = {"metric": metric, "datasets": [{"name": n, "score": score[n], "interactions": inter[n]} for n in score]}
    print(task, len(score), "datasets")
json.dump(out, open(pathlib.Path(__file__).resolve().parent / "paper_results.json", "w"), indent=1)
