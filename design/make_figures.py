"""Draw the landing page's real-data figures as inline SVG fragments.

Reads design/data/bike_*.json (written by design/data/fit_bike.py) and patches
docs/index.html between the FIG markers: the Bike trio and the
three-questions rows (Illustration and Bike).
"""
import json, pathlib, re
root = pathlib.Path(__file__).resolve().parent
d2 = json.load(open(root / "data/bike_default.json"))
d1 = json.load(open(root / "data/bike_depth1.json"))

def path(pts): return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)

def chart(W, H, series, yd, xlabel=None, ticks=(0, 6, 12, 18, 23), pad=(34, 14, 12, 30), legend=None):
    l, r, t, b = pad; pw = W - l - r; ph = H - t - b
    X = lambda h: l + h / 23 * pw; Y = lambda y: t + ph - (y - yd[0]) / (yd[1] - yd[0]) * ph
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    for g in range(5):
        yy = t + ph * g / 4; o.append(f'<line x1="{l}" x2="{W-r}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
    o.append(f'<line x1="{l}" x2="{W-r}" y1="{t+ph}" y2="{t+ph}" class="axis"/>')
    if xlabel:
        for tv in ticks: o.append(f'<text x="{X(tv):.1f}" y="{H-b+16}" class="tick" text-anchor="middle">{tv}</text>')
        o.append(f'<text x="{l+pw/2}" y="{H-2}" class="axl" text-anchor="middle">{xlabel}</text>')
    for cls, vals in series:
        o.append(f'<path d="{path([(X(h), Y(v)) for h, v in enumerate(vals)])}" class="line {cls}"/>')
    for i, (name, cls) in enumerate(legend or []):
        o.append(f'<line x1="{l+8}" x2="{l+26}" y1="{t+10+i*17}" y2="{t+10+i*17}" class="line {cls}"/><text x="{l+32}" y="{t+14+i*17}" class="leg">{name}</text>')
    return "".join(o) + "</svg>"

def heat(W, H, rows, labels, vmax, pad=(34, 14, 6, 26)):
    l, r, t, b = pad; pw = W - l - r; ph = H - t - b; n = len(rows)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    for j, vals in enumerate(rows):
        for h, v in enumerate(vals):
            a = min(1, abs(v) / vmax)
            o.append(f'<rect x="{l+pw*h/24:.1f}" y="{t+ph*j/n:.1f}" width="{pw/24+.6:.1f}" height="{ph/n-1:.1f}" class="{"hp" if v > 0 else "hn"}" opacity="{a*.9+.06:.2f}"/>')
        o.append(f'<text x="{l-4}" y="{t+ph*(j+.5)/n+4:.1f}" class="tick" text-anchor="end">{labels[j]}</text>')
    o.append(f'<text x="{l+pw/2}" y="{H-6}" class="axl" text-anchor="middle">hour of day</text>')
    return "".join(o) + "</svg>"

col = lambda d, key: next(c["values"] for c in d["calm_hr"] if key in c["region"])
yd = (-200, 400)
pairs_hr = [t for t in d2["ga2m_terms"] if len(t) == 2 and "hour" in t]
k = len(pairs_hr)
pair = d2["ga2m_hr_x_workingday"]
wd, off = col(d1, "workingday = True"), col(d1, "workingday = False")
trio = f"""<div class="trio">
 <figure class="mcard"><figcaption><b>GAM</b><span>one curve per feature</span></figcaption>{chart(300, 200, [("s0", d2["gam_hr"])], yd, "hour of day")}<p>The interaction is averaged away.</p></figure>
 <figure class="mcard hi"><figcaption><b>CALM</b><span>one curve per feature, per region</span></figcaption>{chart(300, 200, [("s1", wd), ("s2", off)], yd, "hour of day", legend=[("working day", "s1"), ("day off", "s2")])}<p>The interaction is two curves you can read.</p></figure>
 <figure class="mcard"><figcaption><b>GA²M</b><span>curves plus pairwise heatmaps</span></figcaption>{chart(300, 104, [("s0", d2["ga2m_hr_main"])], yd, pad=(34, 14, 8, 8))}{heat(300, 96, [pair["True"], pair["False"]], ["work", "off"], 230, pad=(34, 14, 6, 26))}<p>The interaction is a surface to decode; one of {k} that involve the hour.</p></figure>
</div>"""



# ---------- three questions: a small chart of the two regional curves, with marks on top ----------
def num(v, dec=0): return ("−" if v < 0 else "+") + f"{abs(v):.{dec}f}"

class Mini:
    def __init__(self, xs, curves, xd, yd, xticks, yticks, xlabel, W=300, H=176, pad=(34, 12, 10, 30)):
        self.xs, self.curves, self.xd, self.yd, self.W, self.H = xs, curves, xd, yd, W, H
        self.l, self.r, self.t, self.b = pad; self.pw, self.ph = W - self.l - self.r, H - self.t - self.b
        self.xticks, self.yticks, self.xlabel = xticks, yticks, xlabel
    def X(self, x): return self.l + (x - self.xd[0]) / (self.xd[1] - self.xd[0]) * self.pw
    def Y(self, y): return self.t + self.ph - (y - self.yd[0]) / (self.yd[1] - self.yd[0]) * self.ph
    def f(self, cls, x):
        xs, ys = self.xs, self.curves[cls]
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x + 1e-9); w = (x - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] * (1 - w) + ys[i + 1] * w
    def seg(self, cls, a, b):
        pts = [(a, self.f(cls, a))] + [(x, y) for x, y in zip(self.xs, self.curves[cls]) if a < x < b] + [(b, self.f(cls, b))]
        return f'<path d="{path([(self.X(x), self.Y(y)) for x, y in pts])}" class="line {cls} hl"/>'
    def dot(self, cls, x): return f'<circle cx="{self.X(x):.1f}" cy="{self.Y(self.f(cls, x)):.1f}" r="4.2" class="dot {cls}"/>'
    def base(self, dim):
        o = [f'<svg viewBox="0 0 {self.W} {self.H}" role="img" class="chart">']
        for v, lab in self.yticks:
            o.append(f'<line x1="{self.l}" x2="{self.W - self.r}" y1="{self.Y(v):.1f}" y2="{self.Y(v):.1f}" class="grid"/><text x="{self.l - 5}" y="{self.Y(v) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{self.l}" x2="{self.W - self.r}" y1="{self.t + self.ph}" y2="{self.t + self.ph}" class="axis"/>')
        for v, lab in self.xticks: o.append(f'<text x="{self.X(v):.1f}" y="{self.H - self.b + 15}" class="tick" text-anchor="middle">{lab}</text>')
        o.append(f'<text x="{self.l + self.pw / 2}" y="{self.H - 2}" class="axl" text-anchor="middle">{self.xlabel}</text>')
        for cls, ys in self.curves.items():
            o.append(f'<path d="{path([(self.X(x), self.Y(y)) for x, y in zip(self.xs, ys)])}" class="line {cls}{" dim" if cls in dim else ""}"/>')
        return o
    def p1(self, cls, x):  # one point, read against the y axis
        y = self.f(cls, x); o = self.base([c for c in self.curves if c != cls])
        o.append(f'<path d="M{self.X(x):.1f},{self.t + self.ph} L{self.X(x):.1f},{self.Y(y):.1f} L{self.l},{self.Y(y):.1f}" class="guide"/>' + self.dot(cls, x))
        return "".join(o) + "</svg>"
    def p2(self, a, b):  # the same step on every curve
        o = self.base(list(self.curves))
        for cls in self.curves: o += [self.seg(cls, a, b), self.dot(cls, a), self.dot(cls, b)]
        return "".join(o) + "</svg>"
    def p3(self, a, b):  # an interval, every curve inside it
        o = self.base(list(self.curves))
        o.insert(1, f'<rect x="{self.X(a):.1f}" y="{self.t}" width="{self.X(b) - self.X(a):.1f}" height="{self.ph}" class="span"/>')
        return "".join(o + [self.seg(cls, a, b) for cls in self.curves]) + "</svg>"

# the three properties, as the paper states them
PROPERTY = {1: 'What is the contribution of each feature to the prediction?', 2: 'How does changing x<sub>i</sub> change the prediction?', 3: 'Is the model globally monotonic increasing with respect to x<sub>i</sub>?'}
sw = lambda cls: f'<i class="sw {cls}"></i>'
def row(n, scope, here, calm_chart, calm_ans, calm_how, chips, ga2m_ans, ga2m_how):
    chips = "".join(f'<span class="chip{" c1" if i == 0 else ""}">{c}</span>' for i, c in enumerate(chips))
    return f"""<div class="prow">
  <div class="pq"><p class="kicker">Property {n} ({scope})</p><h3>{PROPERTY[n]}</h3><p>Here: {here}</p></div>
  <div class="pa calm"><p class="who">CALM</p>{calm_chart}<p class="ans">{calm_ans}</p><p class="how">{calm_how}</p></div>
  <div class="pa ga2m"><p class="who">GA²M</p><div class="chips">{chips}</div><p class="ans no">{ga2m_ans}</p><p class="how">{ga2m_how}</p></div>
</div>"""

# Bike: the hour, working day / day off. The examples: a point (8 am), a step (8 -> 10 am), an interval (5 -> 8 am).
bike = Mini(list(range(24)), {"s1": wd, "s2": off}, (0, 23), yd, [(h, str(h)) for h in (0, 6, 12, 18, 23)],
            [(-200, "−200"), (0, "0"), (200, "200"), (400, "400")], "hour of day")
label = {"feel_temp": "feels-like temp.", "temp": "temperature", "workingday": "working day"}
chips_b = ["hour"] + ["hour × " + label.get(o, o) for o in (next(f for f in t if f != "hour") for t in pairs_hr)]
assert all(wd[h] < wd[h + 1] and off[h] < off[h + 1] for h in (5, 6, 7)), "5 -> 8 am no longer rises on both curves"
props_real = "\n".join([
    row(1, "local", "what does the hour add at 8 am on a working day?",
        bike.p1("s1", 8), f'{num(wd[8])} <small>rentals per hour</small>', "Find the hour on the working-day curve and read the value.",
        chips_b, "No single answer", f"The hour sits in one curve and {k} surfaces. Each surface is shared with another feature, and the model does not say how to split it. SHAP and LIME each split it differently."),
    row(2, "regional", "how does the prediction change when the hour goes from 8 am to 10 am?",
        bike.p2(8, 10), f'{sw("s1")}{num(wd[10] - wd[8])} <small>working day</small><br>{sw("s2")}{num(off[10] - off[8])} <small>day off</small>', "One answer per curve, each exact.",
        chips_b, "It depends on everything else", f"All {k} surfaces move, each by an amount set by another feature. There is no answer until those {k} features are fixed."),
    row(3, "global", "do rentals always rise from 5 am to 8 am?",
        bike.p3(5, 8), "Yes", "Both curves climb there. Two curves checked, and the statement holds for every input.",
        chips_b, "Not by eye", f"The curve climbs, but each of the {k} surfaces could undo it somewhere. All of them have to be checked together."),
])

# Illustration: x3, x2 <= 0 / x2 > 0 (the curves of the hand-drawn trio: 0.3 - 0.9 x^2 and x)
xs = [-1 + i / 60 for i in range(121)]
syn = Mini(xs, {"s1": [0.3 - 0.9 * x * x for x in xs], "s2": xs}, (-1, 1), (-1.1, 1.1), [(-1, "−1"), (0, "0"), (1, "1")], [(-1, "−1"), (0, "0"), (1, "1")], "x₃")
chips_s = ["x₃", "x₃ × x₂"]
props_synth = "\n".join([
    row(1, "local", "what does x₃ add at x₃ = 0.5, when x₂ &gt; 0?",
        syn.p1("s2", 0.5), num(syn.f("s2", 0.5), 1), "Find x₃ on the x₂ &gt; 0 curve and read the value.",
        chips_s, "No single answer", "x₃ sits in a curve and in a surface shared with x₂. The model does not say how to split the surface. SHAP and LIME each split it differently."),
    row(2, "regional", "how does the prediction change when x₃ goes from 0 to 1?",
        syn.p2(0, 1), f'{sw("s2")}{num(syn.f("s2", 1) - syn.f("s2", 0), 1)} <small>when x₂ &gt; 0</small><br>{sw("s1")}{num(syn.f("s1", 1) - syn.f("s1", 0), 1)} <small>when x₂ ≤ 0</small>', "One answer per curve, each exact.",
        chips_s, "It depends on x₂", "The surface moves by an amount set by x₂. With more features, one surface per feature that x₃ interacts with."),
    row(3, "global", "does raising x₃ always raise the prediction?",
        syn.p3(-1, 0), "Only below 0", "Both curves climb up to x₃ = 0. After it, the x₂ ≤ 0 curve turns down.",
        chips_s, "Not by eye", "The curve climbs, but the surface could undo it somewhere. Every row of the surface has to be checked against it."),
])

# ---------- accuracy: three cards on one scale, then the paper's datasets ----------
paper = json.load(open(root / "data/paper_results.json"))
synth_acc = json.load(open(root / "data/synth_accuracy.json"))["rmse"]

def acc_cards(v, dec, note):
    """v: RMSE of gam / calm / ga2m / blackbox. The bars share one scale, the GAM's error being the full width."""
    def card(key, who, hi):
        if key == "gam": how = "the reference"
        else: how = f"{round((1 - v[key] / v['gam']) * 100)}% less error than the GAM"
        return (f'<div class="acard{" hi" if hi else ""}"><p class="who">{who}</p><p class="big">{v[key]:.{dec}f} <small>RMSE</small></p>'
                f'<div class="bar"><i style="width:{v[key] / v["gam"] * 100:.1f}%"></i><b style="left:{v["blackbox"] / v["gam"] * 100:.1f}%"></b></div><p class="how">{how}</p></div>')
    return (f'<div class="acc">{card("gam", "GAM", False)}{card("calm", "CALM", True)}{card("ga2m", "GA²M", False)}</div>'
            f'<p class="cap">{note} The tick marks the black box, {v["blackbox"]:.{dec}f}.</p>')

bs = next(r for r in paper["regression"]["datasets"] if r["name"] == "Bike Sharing")["score"]
acc_real = acc_cards({"gam": bs["gam"], "calm": bs["calm"], "ga2m": bs["eb2m"], "blackbox": bs["blackbox"]}, 1,
                     "Bike Sharing: error on held-out data, lower is better (5 folds, from the paper).")
acc_synth = acc_cards(synth_acc, 2, "Error on held-out data, lower is better (5 folds). The noise puts the best possible value at 0.10.")

def gains(task):
    """Per dataset, what CALM and EB2M gain over the GAM: % lower RMSE, or accuracy points."""
    acc = paper[task]["metric"] == "accuracy"
    g = lambda r, k: (r["score"][k] - r["score"]["gam"]) * 100 if acc else (1 - r["score"][k] / r["score"]["gam"]) * 100
    return sorted(((r["name"], g(r, "calm"), g(r, "eb2m")) for r in paper[task]["datasets"]), key=lambda x: -x[2])

def datasets_chart():
    W, L, R, rh = 880, 170, 760, 22
    blocks = [("Regression", "% lower RMSE than the GAM", gains("regression"), (0, 55), range(0, 51, 10), "%"),
              ("Classification", "accuracy points over the GAM", gains("classification"), (-4, 8), range(-4, 9, 2), "")]
    H = sum(54 + rh * len(b[2]) + 34 for b in blocks)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart wide-chart">']
    y = 0
    for title, xlabel, rows, dom, ticks, unit in blocks:
        X = lambda v: L + (v - dom[0]) / (dom[1] - dom[0]) * (R - L)
        top = y + 44; bot = top + rh * len(rows)
        o.append(f'<text x="0" y="{y + 20}" class="leg blk">{title}</text>')
        if y == 0:
            o.append(f'<circle cx="{R - 150}" cy="{y + 16}" r="5" class="mk-calm"/><text x="{R - 140}" y="{y + 20}" class="leg">CALM</text>'
                     f'<circle cx="{R - 78}" cy="{y + 16}" r="4.5" class="mk-ga2m"/><text x="{R - 68}" y="{y + 20}" class="leg">GA²M (EB²M)</text>')
        for tv in ticks:
            o.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{top}" y2="{bot}" class="{"axis" if tv == 0 else "grid"}"/>'
                     f'<text x="{X(tv):.1f}" y="{bot + 15}" class="tick" text-anchor="middle">{"GAM" if tv == 0 else num(tv).lstrip("+") + unit}</text>')
        o.append(f'<text x="{(L + R) / 2}" y="{bot + 30}" class="axl" text-anchor="middle">{xlabel}</text>')
        for i, (name, c, e) in enumerate(rows):
            cy = top + rh * (i + .5)
            o.append(f'<text x="{L - 12}" y="{cy + 4:.1f}" class="tick row" text-anchor="end">{name}</text>'
                     f'<line x1="{X(c):.1f}" x2="{X(e):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="link"/>'
                     f'<circle cx="{X(e):.1f}" cy="{cy:.1f}" r="4.5" class="mk-ga2m"/><circle cx="{X(c):.1f}" cy="{cy:.1f}" r="5" class="mk-calm"/>'
                     f'<text x="{R + 14}" y="{cy + 4:.1f}" class="tick">{num(c, 1 if unit == "" else 0)}{unit} · {num(e, 1 if unit == "" else 0)}{unit}</text>')
        y = bot + 34 + 10
    return "".join(o) + "</svg>"

def tradeoff_chart():
    W, H, pw, ph, top = 880, 300, 330, 210, 34
    name = {"calm": "CALM", "eb2m": "GA²M (EB²M)", "node": "NODE-GA²M", "gami": "GAMI-Net"}
    mean = lambda xs: sum(xs) / len(xs)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart wide-chart">']
    for left, task, title, dom, ticks in ((60, "regression", "Regression · % lower RMSE than the GAM, mean of 15", (-6, 20), (-5, 0, 5, 10, 15, 20)),
                                          (500, "classification", "Classification · accuracy points over the GAM, mean of 10", (-0.5, 1.5), (-0.5, 0, 0.5, 1, 1.5))):
        ds = paper[task]["datasets"]; acc = paper[task]["metric"] == "accuracy"
        g = lambda r, k: (r["score"][k] - r["score"]["gam"]) * 100 if acc else (1 - r["score"][k] / r["score"]["gam"]) * 100
        X = lambda v: left + (v ** .5) / (110 ** .5) * pw
        Y = lambda v: top + ph - (v - dom[0]) / (dom[1] - dom[0]) * ph
        o.append(f'<text x="{left}" y="18" class="leg blk">{title}</text>')
        for tv in ticks:
            o.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="{"axis" if tv == 0 else "grid"}"/>'
                     f'<text x="{left - 6}" y="{Y(tv) + 4:.1f}" class="tick" text-anchor="end">{("%g" % tv).replace("-", "−")}</text>')
        for tv in (0, 5, 15, 50, 100):
            o.append(f'<text x="{X(tv):.1f}" y="{top + ph + 16}" class="tick" text-anchor="middle">{tv}</text>')
        o.append(f'<text x="{left + pw / 2}" y="{top + ph + 34}" class="axl" text-anchor="middle">pairwise interactions in the model, mean</text>')
        o.append(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="5" class="mk-gam"/><text x="{X(0) + 9:.1f}" y="{Y(0) - 8:.1f}" class="leg">GAM</text>')
        for k in ("gami", "node", "eb2m", "calm"):
            x, yv = X(mean([r["interactions"][k] for r in ds])), Y(mean([g(r, k) for r in ds]))
            end = k in ("node", "calm")
            o.append(f'<circle cx="{x:.1f}" cy="{yv:.1f}" r="{6 if k == "calm" else 4.5}" class="{"mk-calm" if k == "calm" else "mk-ga2m" if k == "eb2m" else "mk-other"}"/>'
                     f'<text x="{x - 10 if end else x + 10:.1f}" y="{yv + 4:.1f}" class="leg{" b" if k == "calm" else ""}" text-anchor="{"end" if end else "start"}">{name[k]}</text>')
    return "".join(o) + "</svg>"

p = root.parent / "docs/index.html"; s = p.read_text()
for tag, frag in (("TRIO-REAL", trio), ("PROPS-SYNTH", "\n" + props_synth + "\n"), ("PROPS-REAL", "\n" + props_real + "\n"),
                  ("ACC-SYNTH", acc_synth), ("ACC-REAL", acc_real), ("MORE-DATASETS", datasets_chart()), ("MORE-TRADEOFF", tradeoff_chart())):
    pat = re.compile(rf"(<!-- FIG:{tag} -->).*?(<!-- /FIG:{tag} -->)", re.S)
    assert pat.search(s), tag
    s = pat.sub(lambda m: m.group(1) + frag + m.group(2), s)
p.write_text(s); print("patched", p, "|", k, "pairs with the hour |", num(wd[8]), num(wd[10] - wd[8]), num(off[10] - off[8]))
