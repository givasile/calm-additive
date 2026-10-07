"""Draw the landing page's figures, one SVG file each, in the style of a pgfplots axis.

Reads design/data/*.json (written by fit_bike.py and paper_tables.py) and writes
docs/figures/<name>.svg. The page shows them with <img>, so each file carries its
own style, in the page's paper colours. The page itself is written by hand.

Style: a closed hairline frame, tick marks pointing inward on all four sides,
no grid on the curve plots, math-italic axis labels, a boxed legend inside the
axes, thin lines and small markers.
"""
import json, pathlib, re
root = pathlib.Path(__file__).resolve().parent
d2 = json.load(open(root / "data/bike_default.json"))
d1 = json.load(open(root / "data/bike_depth1.json"))
paper = json.load(open(root / "data/paper_results.json"))

TK = 3.5          # tick length, in viewBox units
FS = 14           # chart font size in px, the page sets it; 1 unit is about 1 px
FS_BIG = 17       # the three-figure row renders smaller than the other charts, so its text is set larger
def path(pts): return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
def num(v, dec=0): return ("−" if v < 0 else "+") + f"{abs(v):.{dec}f}"
def var(name, sub=None):
    """A math-italic variable with an optional subscript, like $x_3$."""
    return f'<tspan class="it">{name}</tspan>' + (f'<tspan dy="3" font-size="75%">{sub}</tspan><tspan dy="-3">&#8203;</tspan>' if sub is not None else "")
def interp(xs, ys, x):
    i = max(j for j in range(len(xs) - 1) if xs[j] <= x + 1e-9); w = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1 - w) + ys[i + 1] * w
def twidth(s, fs=FS): return 0.52 * fs * len(re.sub(r"<[^>]+>", "", s))   # a rough width for Times


class Axes:
    def __init__(self, W, H, xd, yd, pad, big=False):
        self.W, self.H, self.xd, self.yd = W, H, xd, yd
        self.l, self.r, self.t, self.b = pad
        self.pw, self.ph = W - self.l - self.r, H - self.t - self.b
        self.fs = FS_BIG if big else FS
        self.o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart{" big" if big else ""}">']
    def X(self, x): return self.l + (x - self.xd[0]) / (self.xd[1] - self.xd[0]) * self.pw
    def Y(self, y): return self.t + self.ph - (y - self.yd[0]) / (self.yd[1] - self.yd[0]) * self.ph
    def add(self, s): self.o.append(s)
    def frame(self):
        self.add(f'<rect x="{self.l}" y="{self.t}" width="{self.pw}" height="{self.ph}" class="frame"/>')
    def xticks(self, ticks, labels=True, grid=False):
        y0, y1 = self.t + self.ph, self.t
        for v, lab in ticks:
            x = self.X(v)
            if grid: self.add(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" class="grid"/>')
            self.add(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y0 - TK}" class="tk"/><line x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y1 + TK}" class="tk"/>')
            if labels: self.add(f'<text x="{x:.1f}" y="{y0 + self.fs + 1}" class="tick" text-anchor="middle">{lab}</text>')
    def yticks(self, ticks, labels=True, grid=False):
        x0, x1 = self.l, self.l + self.pw
        for v, lab in ticks:
            y = self.Y(v)
            if grid: self.add(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
            self.add(f'<line x1="{x0}" x2="{x0 + TK}" y1="{y:.1f}" y2="{y:.1f}" class="tk"/><line x1="{x1}" x2="{x1 - TK}" y1="{y:.1f}" y2="{y:.1f}" class="tk"/>')
            if labels: self.add(f'<text x="{x0 - 5}" y="{y + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    def xlabel(self, s): self.add(f'<text x="{self.l + self.pw / 2:.1f}" y="{self.H - 3}" class="axl" text-anchor="middle">{s}</text>')
    def ylabel(self, s): self.add(f'<text transform="translate(12,{self.t + self.ph / 2:.1f}) rotate(-90)" class="axl" text-anchor="middle">{s}</text>')
    def line(self, cls, pts, extra=""): self.add(f'<path d="{path([(self.X(x), self.Y(y)) for x, y in pts])}" class="line {cls}{extra}"/>')
    def legend(self, entries, pos="nw", box=True):
        """A legend inside the axes: a line sample and a name per entry, boxed or not."""
        lh, padx, pady, sample = self.fs + 3, 6, 4, 16
        w = padx + sample + 5 + max(twidth(n, self.fs) for n, _ in entries) + padx; h = pady * 2 + lh * len(entries)
        x = self.l + 6 if "w" in pos else self.l + self.pw - 6 - w
        y = self.t + 6 if "n" in pos else self.t + self.ph - 6 - h
        if box: self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" class="legbox"/>')
        for i, (name, cls) in enumerate(entries):
            cy = y + pady + lh * i + lh / 2
            self.add(f'<line x1="{x + padx:.1f}" x2="{x + padx + sample:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="line {cls}"/>'
                     f'<text x="{x + padx + sample + 5:.1f}" y="{cy + 4.5:.1f}" class="leg">{name}</text>')
    def svg(self): return "".join(self.o) + "</svg>"


def curves(W, H, xs, series, xd, yd, xticks, yticks, xlabel, ylabel, pad, legend=None, xlabels=True, labels=()):
    """labels: (text, x, y, anchor) nodes placed on the plot, for curves named directly instead of by a legend."""
    a = Axes(W, H, xd, yd, pad, big=True)
    a.yticks(yticks, labels=False, grid=True)
    for cls, ys in series: a.line(cls, zip(xs, ys))
    a.frame(); a.xticks(xticks, labels=xlabels); a.yticks(yticks)
    if xlabel: a.xlabel(xlabel)
    if ylabel: a.ylabel(ylabel)
    if legend: a.legend(legend, box=False)
    for text, x, y, anchor in labels: a.add(f'<text x="{a.X(x):.1f}" y="{a.Y(y):.1f}" class="leg" text-anchor="{anchor}">{text}</text>')
    return a.svg()

def heat(W, H, rows, xd, xticks, ylabel, yticks, vmax, vlabel, pad, uid, xlabel=None):
    """A matrix plot: one cell per (row, column), a diverging scale through the page colour, a colorbar at the right."""
    a = Axes(W, H, xd, (0, 1), pad, big=True)
    cells(a, rows, vmax); a.frame(); a.xticks(xticks, labels=xlabel is not None)
    for v, lab in yticks: a.add(f'<text x="{a.l - 5}" y="{a.Y(v) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    if xlabel: a.xlabel(xlabel)
    if ylabel: a.ylabel(ylabel)
    colorbar(a, uid, vlabel)
    return a.svg()

def colorbar(a, uid, vlabel):
    """A vertical colorbar beside the axes: the positive hue fading into the page, then the negative one."""
    x, w, y0, y1 = a.l + a.pw + 8, 7, a.t, a.t + a.ph; ym = (y0 + y1) / 2
    a.add(f'<defs><linearGradient id="cbp{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--s2);stop-opacity:.96"/><stop offset="1" style="stop-color:var(--s2);stop-opacity:.06"/></linearGradient>'
          f'<linearGradient id="cbn{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--s1);stop-opacity:.06"/><stop offset="1" style="stop-color:var(--s1);stop-opacity:.96"/></linearGradient></defs>'
          f'<rect x="{x}" y="{y0}" width="{w}" height="{ym - y0:.1f}" fill="url(#cbp{uid})"/><rect x="{x}" y="{ym:.1f}" width="{w}" height="{y1 - ym:.1f}" fill="url(#cbn{uid})"/>'
          f'<rect x="{x}" y="{y0}" width="{w}" height="{a.ph}" class="frame"/>')
    for yy, lab in ((y0, "+" + vlabel), (ym, "0"), (y1, "−" + vlabel)):
        a.add(f'<line x1="{x + w}" x2="{x + w + 2.5}" y1="{yy:.1f}" y2="{yy:.1f}" class="tk"/><text x="{x + w + 4}" y="{yy + 3.5:.1f}" class="cbt">{lab}</text>')

def cells(a, rows, vmax):
    n, m = len(rows), len(rows[0]); cw, ch = a.pw / m, a.ph / n
    for j, vals in enumerate(rows):
        for i, v in enumerate(vals):
            op = min(1, abs(v) / vmax) * .9 + .06
            a.add(f'<rect x="{a.l + cw * i:.2f}" y="{a.t + ch * j:.2f}" width="{cw:.2f}" height="{ch:.2f}" shape-rendering="crispEdges" class="{"hp" if v > 0 else "hn"}" opacity="{op:.2f}"/>')


class Mini:
    """The property charts: the regional curves, with a reading drawn on top."""
    def __init__(self, xs, curves, xd, yd, xticks, yticks, xlabel, ylabel, names, pad, W=300, H=176):
        self.xs, self.curves, self.xd, self.yd, self.W, self.H, self.pad = xs, curves, xd, yd, W, H, pad
        self.xticks, self.yticks, self.xlabel, self.ylabel, self.names = xticks, yticks, xlabel, ylabel, dict(names)
    def f(self, cls, x): return interp(self.xs, self.curves[cls], x)
    def base(self, dim):
        a = Axes(self.W, self.H, self.xd, self.yd, self.pad)
        for cls, ys in self.curves.items(): a.line(cls, zip(self.xs, ys), " dim" if cls in dim else "")
        return a
    def finish(self, a):
        a.frame(); a.xticks(self.xticks); a.yticks(self.yticks); a.xlabel(self.xlabel); a.ylabel(self.ylabel)
        return a.svg()
    def seg(self, a, cls, lo, hi):
        pts = [(lo, self.f(cls, lo))] + [(x, y) for x, y in zip(self.xs, self.curves[cls]) if lo < x < hi] + [(hi, self.f(cls, hi))]
        a.line(cls, pts, " hl")
    def dot(self, a, cls, x): a.add(f'<circle cx="{a.X(x):.1f}" cy="{a.Y(self.f(cls, x)):.1f}" r="3.2" class="dot {cls}"/>')
    def p1(self, cls, x, label_at):  # one point, read against the y axis; the curve is named, like a pgfplots node on it
        a = self.base([c for c in self.curves if c != cls]); y = self.f(cls, x)
        a.add(f'<path d="M{a.X(x):.1f},{a.t + a.ph} L{a.X(x):.1f},{a.Y(y):.1f} L{a.l},{a.Y(y):.1f}" class="guide"/>'); self.dot(a, cls, x)
        lx, ly, anchor = label_at
        a.add(f'<text x="{a.X(lx):.1f}" y="{a.Y(ly):.1f}" class="leg" text-anchor="{anchor}">{self.names[cls]}</text>')
        return self.finish(a)
    def p2(self, lo, hi):  # the same step on every curve
        a = self.base(list(self.curves))
        for cls in self.curves: self.seg(a, cls, lo, hi); self.dot(a, cls, lo); self.dot(a, cls, hi)
        return self.finish(a)
    def p3(self, lo, hi):  # an interval, every curve inside it
        a = self.base(list(self.curves))
        a.o.insert(1, f'<rect x="{a.X(lo):.1f}" y="{a.t}" width="{a.X(hi) - a.X(lo):.1f}" height="{a.ph}" class="span"/>')
        for cls in self.curves: self.seg(a, cls, lo, hi)
        return self.finish(a)


def ga2m_panel(xs, main, rows, xd, yd, xticks, yticks, xlabel, ylabel_top, ylabel_bot, row_labels, vmax, vlabel, pad, uid, mark, W=300, H=176, hh=46, gap=8):
    """One SVG: the GA²M curve of the feature on top, its surface with a partner feature below.
    mark: ("p1", x, (r0, r1)) a point on the curve and a dashed line through the rows r0..r1 (fractions) of the surface;
          ("p2", a, b) a step on the curve and two dashed lines through the surface;
          ("p3", a, b) an interval on both."""
    l, r, t, b = pad
    top = Axes(W, H, xd, yd, (l, r, t, b + hh + gap)); bot = Axes(W, H, xd, (0, 1), (l, r, H - b - hh, b))
    kind, *args = mark
    if kind == "p3": top.add(f'<rect x="{top.X(args[0]):.1f}" y="{top.t}" width="{top.X(args[1]) - top.X(args[0]):.1f}" height="{top.ph}" class="span"/>')
    top.line("s0", zip(xs, main))
    if kind == "p1":
        x, (r0, r1) = args; y = interp(xs, main, x)
        top.add(f'<path d="M{top.X(x):.1f},{top.t + top.ph} L{top.X(x):.1f},{top.Y(y):.1f} L{top.l},{top.Y(y):.1f}" class="guide"/>'
                f'<circle cx="{top.X(x):.1f}" cy="{top.Y(y):.1f}" r="3.2" class="dot s0"/>')
    else:
        a_, b_ = args
        pts = [(a_, interp(xs, main, a_))] + [(x, y) for x, y in zip(xs, main) if a_ < x < b_] + [(b_, interp(xs, main, b_))]
        top.line("s0", pts, " hl")
        if kind == "p2":
            for x in (a_, b_): top.add(f'<circle cx="{top.X(x):.1f}" cy="{top.Y(interp(xs, main, x)):.1f}" r="3.2" class="dot s0"/>')
    top.frame(); top.xticks(xticks, labels=False); top.yticks(yticks); top.ylabel(ylabel_top)
    cells(bot, rows, vmax)
    if kind == "p1":
        x, (r0, r1) = args
        bot.add(f'<line x1="{bot.X(x):.1f}" x2="{bot.X(x):.1f}" y1="{bot.Y(r0):.1f}" y2="{bot.Y(r1):.1f}" class="guide hm"/>')
    elif kind == "p2":
        for x in args: bot.add(f'<line x1="{bot.X(x):.1f}" x2="{bot.X(x):.1f}" y1="{bot.t}" y2="{bot.t + bot.ph}" class="guide hm"/>')
    else:
        bot.add(f'<rect x="{bot.X(args[0]):.1f}" y="{bot.t}" width="{bot.X(args[1]) - bot.X(args[0]):.1f}" height="{bot.ph}" class="span hm"/>')
        for x in args: bot.add(f'<line x1="{bot.X(x):.1f}" x2="{bot.X(x):.1f}" y1="{bot.t}" y2="{bot.t + bot.ph}" class="guide hm"/>')  # the span's edges, over the cells
    bot.frame(); bot.xticks(xticks); bot.xlabel(xlabel)
    for v, lab in row_labels: bot.add(f'<text x="{bot.l - 5}" y="{bot.Y(v) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    if ylabel_bot: bot.add(f'<text transform="translate(12,{bot.t + bot.ph / 2:.1f}) rotate(-90)" class="axl" text-anchor="middle">{ylabel_bot}</text>')
    colorbar(bot, uid, vlabel)
    return "".join(top.o + bot.o[1:]) + "</svg>"


# ---------- illustration: x3 has two curves, 0.3 - 0.9 x^2 when x2 <= 0 and x when x2 > 0 ----------
xs = [-1 + i / 60 for i in range(121)]
c1, c2 = [0.3 - 0.9 * x * x for x in xs], list(xs)
gam_s = [(u + v) / 2 for u, v in zip(c1, c2)]
x3, x2, fx3 = var("x", 3), var("x", 2), var("y")
st = dict(xd=(-1, 1), yd=(-1.1, 1.1), xticks=[(-1, "−1"), (0, "0"), (1, "1")], yticks=[(-1, "−1"), (0, "0"), (1, "1")])
leg_s = [(x2 + " ≤ 0", "s1"), (x2 + " &gt; 0", "s2")]
PAD_S = (44, 12, 10, 32)
PAD_T = (48, 12, 10, 38)   # the three-figure row
synth_gam = curves(300, 200, xs, [("s0", gam_s)], xlabel=x3, ylabel=fx3, pad=PAD_T, **st)
synth_calm = curves(300, 200, xs, [("s1", c1), ("s2", c2)], xlabel=x3, ylabel=fx3, pad=PAD_T, legend=leg_s, **st)
synth_ga2m_curve = curves(300, 104, xs, [("s0", gam_s)], xlabel=None, ylabel=fx3, pad=(48, 48, 8, 8), xlabels=False, **st)   # the GA²M column keeps room for the colorbar
grid2 = [1 - (j + .5) / 4 for j in range(8)]   # x2 from top (+1) to bottom (−1)
surf_s = [[v2 - gam_s[i] for i, v2 in enumerate(c2 if r > 0 else c1)][::4] for r in grid2]
synth_heat = heat(300, 96, surf_s,
                  (-1, 1), [(-1, "−1"), (0, "0"), (1, "1")], x2, [(0.98, "1"), (0.02, "−1")], 0.8, "0.8", (48, 48, 6, 38), "s", xlabel=x3)
syn = Mini(xs, {"s1": c1, "s2": c2}, xlabel=x3, ylabel=var("y"), names=[(cls, n) for n, cls in leg_s], pad=PAD_S, **st)
props_synth = {"P1-SYNTH": syn.p1("s2", 0.5, (0.9, 0.55, "end")), "P2-SYNTH": syn.p2(0, 1), "P3-SYNTH": syn.p3(-1, 0)}
ga_s = dict(xs=xs, main=gam_s, rows=surf_s, xlabel=x3, ylabel_top=var("y"), ylabel_bot=x2, row_labels=[(0.98, "1"), (0.02, "−1")],
            vmax=0.8, vlabel="0.8", pad=(44, 44, 10, 32), uid="gs", **st)
ga2m_synth = {"GA-P1-SYNTH": ga2m_panel(mark=("p1", 0.5, (0.5, 1)), **ga_s), "GA-P2-SYNTH": ga2m_panel(mark=("p2", 0, 1), **ga_s), "GA-P3-SYNTH": ga2m_panel(mark=("p3", -1, 0), **ga_s)}
surf_at = lambda x: interp(xs, [u - v for u, v in zip(c2, gam_s)], x)
print("illustration, GA²M: curve at 0.5", num(interp(xs, gam_s, 0.5), 2), "| surface at 0.5, x2 > 0", num(surf_at(0.5), 2),
      "| curve step 0 -> 1", num(interp(xs, gam_s, 1) - interp(xs, gam_s, 0), 2), "| surface step, x2 > 0", num(surf_at(1) - surf_at(0), 2))

# ---------- Bike Sharing: the hour, on a working day and on a day off ----------
col = lambda d, key: next(c["values"] for c in d["calm_hr"] if key in c["region"])
wd, off = col(d1, "workingday = True"), col(d1, "workingday = False")
pair = d2["ga2m_hr_x_workingday"]; k = len([t for t in d2["ga2m_terms"] if len(t) == 2 and "hour" in t])
hours = list(range(24))
bt = dict(xd=(0, 23), yd=(-200, 400), xticks=[(h, str(h)) for h in (0, 6, 12, 18, 23)], yticks=[(-200, "−200"), (0, "0"), (200, "200"), (400, "400")])
leg_b = [("working day", "s1"), ("day off", "s2")]
PAD_B = (56, 12, 10, 32)
PAD_BT = (66, 12, 10, 38)   # the three-figure row
real_gam = curves(300, 200, hours, [("s0", d2["gam_hr"])], xlabel="hour of day", ylabel="rentals per hour", pad=PAD_BT, **bt)
real_calm = curves(300, 200, hours, [("s1", wd), ("s2", off)], xlabel="hour of day", ylabel="rentals per hour", pad=PAD_BT,
                   labels=[("working day", 17.3, 335, "middle"), ("day off", 12.5, 190, "middle")], **bt)
real_ga2m_curve = curves(300, 104, hours, [("s0", d2["ga2m_hr_main"])], xlabel=None, ylabel="rentals / h", pad=(66, 48, 8, 8), xlabels=False, **bt)
real_heat = heat(300, 96, [pair["True"], pair["False"]], (-0.5, 23.5), [(h, str(h)) for h in (0, 6, 12, 18, 23)], None,
                 [(0.75, "work"), (0.25, "off")], 230, "230", (66, 48, 6, 38), "b", xlabel="hour of day")
bike = Mini(hours, {"s1": wd, "s2": off}, xlabel="hour of day", ylabel="rentals per hour", names=[(cls, n) for n, cls in leg_b], pad=PAD_B, **bt)
assert all(wd[h] < wd[h + 1] and off[h] < off[h + 1] for h in (5, 6, 7)), "5 -> 8 am no longer rises on both curves"
props_real = {"P1-REAL": bike.p1("s1", 8, (9.2, 330, "start")), "P2-REAL": bike.p2(8, 10), "P3-REAL": bike.p3(5, 8)}
main_b = d2["ga2m_hr_main"]
ga_b = dict(xs=hours, main=main_b, rows=[pair["True"], pair["False"]], xlabel="hour of day", ylabel_top="rentals / h", ylabel_bot=None,
            row_labels=[(0.75, "work"), (0.25, "off")], vmax=230, vlabel="230", pad=(56, 44, 10, 32), uid="gb", **{**bt, "xd": (-0.5, 23.5)})
ga2m_real = {"GA-P1-REAL": ga2m_panel(mark=("p1", 8, (0.5, 1)), **ga_b), "GA-P2-REAL": ga2m_panel(mark=("p2", 8, 10), **ga_b), "GA-P3-REAL": ga2m_panel(mark=("p3", 5, 8), **ga_b)}
print("Bike, GA²M: curve at 8", num(main_b[8]), "| working-day surface at 8", num(pair["True"][8]),
      "| curve step 8 -> 10", num(main_b[10] - main_b[8]), "| working-day surface step", num(pair["True"][10] - pair["True"][8]))

# ---------- the paper's datasets: what CALM and the GA²M gain over the GAM ----------
def gains(task):
    acc = paper[task]["metric"] == "accuracy"
    g = lambda r, key: (r["score"][key] - r["score"]["gam"]) * 100 if acc else (1 - r["score"][key] / r["score"]["gam"]) * 100
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
        if y == 0:  # a boxed legend above the axis, and the header of the number column
            bx, bw = R - 206, 206
            o.append(f'<rect x="{bx}" y="{y + 4}" width="{bw}" height="24" class="legbox"/>'
                     f'<circle cx="{bx + 14}" cy="{y + 16}" r="3.6" class="mk-calm"/><text x="{bx + 24}" y="{y + 20.5}" class="leg">CALM</text>'
                     f'<circle cx="{bx + 84}" cy="{y + 16}" r="3.3" class="mk-ga2m"/><text x="{bx + 94}" y="{y + 20.5}" class="leg">GA²M (EB²M)</text>'
                     f'<text x="{R + 14}" y="{y + 20.5}" class="tick">CALM · GA²M</text>')
        o.append(f'<line x1="{R + 14}" x2="{W}" y1="{top - 1}" y2="{top - 1}" class="rule"/>')
        for tv in ticks:
            if tv == 0: o.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{top}" y2="{bot}" class="axis"/>')
            else: o.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{top}" y2="{bot}" class="grid"/>')
            o.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{bot}" y2="{bot - TK}" class="tk"/><line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{top}" y2="{top + TK}" class="tk"/>'
                     f'<text x="{X(tv):.1f}" y="{bot + 15}" class="tick" text-anchor="middle">{"GAM" if tv == 0 else num(tv).lstrip("+") + unit}</text>')
        o.append(f'<text x="{(L + R) / 2}" y="{bot + 30}" class="axl" text-anchor="middle">{xlabel}</text>')
        for i, (name, c, e) in enumerate(rows):
            cy = top + rh * (i + .5)
            o.append(f'<text x="{L - 12}" y="{cy + 4:.1f}" class="tick row" text-anchor="end">{name}</text>'
                     f'<line x1="{X(c):.1f}" x2="{X(e):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="link"/>'
                     f'<circle cx="{X(e):.1f}" cy="{cy:.1f}" r="3.3" class="mk-ga2m"/><circle cx="{X(c):.1f}" cy="{cy:.1f}" r="3.6" class="mk-calm"/>'
                     f'<text x="{R + 14}" y="{cy + 4:.1f}" class="tick">{num(c, 1 if unit == "" else 0)}{unit} · {num(e, 1 if unit == "" else 0)}{unit}</text>')
        o.append(f'<rect x="{L}" y="{top}" width="{R - L}" height="{bot - top}" class="frame"/>')
        y = bot + 34 + 10
    return "".join(o) + "</svg>"

figs = {"SYNTH-GAM": synth_gam, "SYNTH-CALM": synth_calm, "SYNTH-GA2M-CURVE": synth_ga2m_curve, "SYNTH-GA2M-HEAT": synth_heat,
        "REAL-GAM": real_gam, "REAL-CALM": real_calm, "REAL-GA2M-CURVE": real_ga2m_curve, "REAL-GA2M-HEAT": real_heat,
        **props_synth, **props_real, **ga2m_synth, **ga2m_real, "DATASETS": datasets_chart()}

CSS = """<style>
  svg { --bg: #f5eedc; --fg: #2a2520; --muted: #6b6155; --line: #dccfb3; --s1: #2b5c8a; --s2: #b85c25; font-family: "Times New Roman", Times, "Liberation Serif", serif; font-size: 14px; }
  svg.big { font-size: 17px; }
  .grid { stroke: var(--line); stroke-width: 1; } .axis { stroke: var(--fg); stroke-width: .9; } .rule { stroke: var(--line); stroke-width: 1; }
  .frame { fill: none; stroke: var(--fg); stroke-width: .9; } .tk { stroke: var(--fg); stroke-width: .9; }
  .legbox { fill: var(--bg); stroke: var(--fg); stroke-width: .7; } .it { font-style: italic; }
  .tick, .axl, .leg { fill: var(--fg); } .leg.blk, .leg.b { font-weight: bold; } .cbt { fill: var(--fg); font-size: 11px; } svg.big .cbt { font-size: 12.5px; }
  .line { fill: none; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; } .line.dim { opacity: .3; } .line.hl { stroke-width: 2.4; }
  .s0 { stroke: var(--muted); } .s1 { stroke: var(--s1); } .s2 { stroke: var(--s2); }
  .hp { fill: var(--s2); } .hn { fill: var(--s1); }
  .dot { stroke: var(--bg); stroke-width: 1.2; } .dot.s0 { fill: var(--muted); } .dot.s1 { fill: var(--s1); } .dot.s2 { fill: var(--s2); }
  .guide { fill: none; stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; } .guide.hm { stroke: var(--fg); }
  .span { fill: var(--fg); opacity: .08; }
  .link { stroke: var(--muted); stroke-width: 1.2; opacity: .6; }
  .mk-calm { fill: var(--s1); stroke: var(--bg); stroke-width: 1.2; } .mk-ga2m { fill: var(--bg); stroke: var(--s2); stroke-width: 1.7; }
</style>"""

out = root.parent / "docs/figures"; out.mkdir(exist_ok=True)
for tag, svg in figs.items():
    w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups()
    head, body = svg.split(">", 1)
    head = head.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ', 1).replace(' role="img"', "")
    (out / f"{tag.lower()}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + head + ">\n" + CSS + "\n" + body + "\n")
print("wrote", len(figs), "figures to", out, "|", k, "pairs with the hour | the answers on the page:", num(wd[8]), num(wd[10] - wd[8]), num(off[10] - off[8]),
      num(syn.f("s2", 0.5), 1), num(syn.f("s2", 1) - syn.f("s2", 0), 1), num(syn.f("s1", 1) - syn.f("s1", 0), 1))
