"""Plot paired CIFAR-100 training results and validation-selected score tradeoffs.

Run from the article or reader plotting directory with numpy, matplotlib,
and Pillow installed. Inputs are recorded results from five ER histories.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "new_experiments/cross_group_supervision"
OUTPUT = ROOT / "figures"
SEEDS = list(range(3101, 3106))
TARGETS = [5, 10, 15, 20]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"

with (SOURCE / "score_analysis/unadjusted_results.csv").open() as handle:
    unadjusted = {(row["model"], int(row["seed"])): float(row["new_pct"])
                  for row in csv.DictReader(handle) if row["images"] == "test"}
with (SOURCE / "score_analysis/offset_results.csv").open() as handle:
    offsets = {(row["model"], int(row["seed"]), int(row["target_pct"])): row
               for row in csv.DictReader(handle)}

plt.rcParams.update({
    "font.family": "Comic Sans MS",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig = plt.figure(figsize=(4.72, 2.84))
left = fig.add_axes([.105, .245, .36, .575])
right = fig.add_axes([.62, .245, .36, .575])
fig.text(.105, .965, "a  Incoming supervision", weight="bold", fontsize=9, va="top")
fig.text(.62, .965, "b  Score adjustment", weight="bold", fontsize=9, va="top")

# Connecting lines retain each history's paired incoming-loss comparison.
for group, (first, second, label, color, marker, style) in enumerate([
    ("usual_er", "current_mask", "Old replay", BLUE, "o", "-"),
    ("mixed_full", "mixed_restricted", "Mixed replay", "#AD4B88", "s", "--"),
]):
    values = np.array([[unadjusted[name, seed] for seed in SEEDS]
                       for name in (first, second)]).T
    pale = np.array(to_rgb(color)) * .5 + .5
    positions = np.array([0, 1]) + (-.035 if group == 0 else .035)
    for shift, pair in zip(np.linspace(-.04, .04, len(SEEDS)), values):
        left.plot(positions + shift, pair, color=pale, lw=.6, ls=style,
                  marker=marker, ms=2.5, mew=.45, zorder=2)
    left.plot(positions, values.mean(axis=0), color=color, lw=1.45, ls=style,
              marker=marker, ms=5, label=label, zorder=4)
left.set(xlim=(-.22, 1.22), ylim=(-2, 86), yticks=[0, 25, 50, 75])
left.set_xticks([0, 1], ["All 100\nclasses", "Current 10\nclasses"])
left.set_xlabel("Classes in incoming loss", labelpad=4)
left.set_ylabel("New-class accuracy (%)", labelpad=3)
left.legend(loc="lower left", bbox_to_anchor=(.015, .045), frameon=False,
            borderpad=0, labelspacing=.3, handlelength=1.6,
            handletextpad=.5, fontsize=8)

# Targets are selected on validation images and evaluated on separate images.
means = []
for name, label, color, marker, style in [
    ("usual_er", "100-way incoming CE", BLUE, "o", "-"),
    ("current_mask", "10-way incoming CE", ORANGE, "^", "--"),
    ("old_blend05", "95/5 incoming CE", GREEN, "s", ":"),
]:
    values = np.array([[[float(offsets[name, seed, target][key])
                         for key in ("test_old_pct", "test_new_pct")]
                        for target in TARGETS] for seed in SEEDS])
    mean = values.mean(axis=0)
    means.append(mean)
    right.plot(mean[:, 0], mean[:, 1], color=color, marker=marker, ms=4,
               ls=style, lw=1.3, label=label, zorder=4)
    pale = np.array(to_rgb(color)) * .65 + .35
    right.scatter(values[:, -1, 0], values[:, -1, 1], marker=marker, s=12,
                  facecolor="white", edgecolor=pale, linewidth=.65, zorder=2)

# One label identifies the three nearby mean points at a shared target.
means = np.array(means)
for index, target in enumerate(TARGETS):
    point = means[:, index, :].mean(axis=0)
    label_y = max(means[:, index, 1]) + (4 if target != 20 else 17)
    right.annotate(f"q={target}", (point[0], label_y), ha="center", va="bottom",
                   fontsize=8, color="#444444")
right.set(xlim=(6, 26), ylim=(-2, 86), yticks=[0, 25, 50, 75], xticks=[8, 16, 24])
right.set_xlabel("Test old-class accuracy (%)", labelpad=4)
right.legend(loc="lower left", frameon=False, borderpad=0, labelspacing=.3,
             handlelength=1.6, handletextpad=.5, fontsize=8,
             bbox_to_anchor=(0, .015))
fig.text(.62, .875, "Validation requires q + 3 pp", ha="left", va="bottom", fontsize=8)

for axis in (left, right):
    axis.grid(axis="y", color="#dddddd", linewidth=.5)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.5, width=.65, pad=2)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for item in fig.findobj(matplotlib.text.Text):
    if item.get_visible() and item.get_text():
        extent = item.get_window_extent(renderer)
        if (extent.x0 < 0 or extent.y0 < 0 or extent.x1 > fig.bbox.width
                or extent.y1 > fig.bbox.height):
            outside.append(item.get_text())
if outside:
    raise ValueError(f"Text outside the figure: {outside}")

OUTPUT.mkdir(exist_ok=True)
for suffix in ["png", "pdf", "svg", "eps"]:
    fig.savefig(OUTPUT / f"supervision_comparison.{suffix}", dpi=300,
                facecolor="white")
plt.close(fig)
with Image.open(OUTPUT / "supervision_comparison.png") as preview:
    preview.convert("L").save(OUTPUT / "supervision_comparison_gray.png")
print("Plotted paired histories and four validation-selected targets.")
