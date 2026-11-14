"""Plot the fixed-history loss comparison and held-out old/new trade-offs."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures"
SOURCE = ROOT / "new_experiments/cross_group_supervision"
OUTPUT.mkdir(exist_ok=True)

summary = json.loads((SOURCE / "summary.json").read_text())
with (SOURCE / "score_analysis/unadjusted_results.csv").open() as handle:
    raw_rows = list(csv.DictReader(handle))
with (SOURCE / "score_analysis/offset_results.csv").open() as handle:
    adjusted_rows = list(csv.DictReader(handle))

seeds = list(range(3101, 3106))
targets = [5, 10, 15, 20]
raw_lookup = {
    (row["model"], int(row["seed"])): row
    for row in raw_rows if row["images"] == "test"
}
adjusted_lookup = {
    (row["model"], int(row["seed"]), int(row["target_pct"])): row
    for row in adjusted_rows
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})

fig = plt.figure(figsize=(4.72, 6.45))

# A. Each seed supplies one ER starting state for every compared branch.
fig.text(0.04, 0.975, "A   Change the final update from the same ER history",
         weight="bold", fontsize=8.8)
diagram = fig.add_axes([0.025, 0.815, 0.95, 0.14])
diagram.axis("off")


def box(x, y, width, height, label):
    diagram.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=.008",
        facecolor="#f5f5f5", edgecolor="#888888", linewidth=0.65,
    ))
    diagram.text(x + width / 2, y + height / 2, label,
                 ha="center", va="center", fontsize=7.7)


def arrow(start, end):
    diagram.annotate("", xy=end, xytext=start,
                     arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.7})


box(0.015, 0.32, 0.22, 0.36, "One ER model\nbefore final task")
box(0.34, 0.60, 0.35, 0.36, "Old-only replay\nThree incoming losses")
box(0.34, 0.04, 0.35, 0.36, "Mixed replay\nTwo incoming losses")
box(0.80, 0.32, 0.19, 0.36, "Predict all\n100 classes")
arrow((0.235, 0.50), (0.34, 0.78))
arrow((0.235, 0.50), (0.34, 0.22))
arrow((0.69, 0.78), (0.80, 0.50))
arrow((0.69, 0.22), (0.80, 0.50))
fig.text(0.5, 0.795, "Train paired copies for 70 epochs in each of five histories.",
         ha="center", fontsize=7.8)

# B. Full 10,000-image test results, retaining all paired training histories.
fig.text(0.04, 0.748, "B   Replay composition changes the incoming-loss effect",
         weight="bold", fontsize=8.8)
loss_axis = fig.add_axes([0.135, 0.485, 0.825, 0.20])
blue, orange, green = "#0072B2", "#D55E00", "#009E73"
for index, (left, right, label, marker, color) in enumerate([
    ("usual_er", "current_mask", "Old-only replay", "o", blue),
    ("mixed_full", "mixed_restricted", "Mixed replay", "s", orange),
]):
    values = np.array([
        summary["methods"][name]["new_all100_pct"]["values"]
        for name in (left, right)
    ]).T
    x = np.array([0, 1]) + (-0.015 if index == 0 else 0.015)
    light = np.array(to_rgb(color)) * 0.35 + 0.65
    for row in values:
        loss_axis.plot(x, row, color=light, linewidth=0.6, marker=marker,
                       markersize=2.4)
    loss_axis.plot(x, values.mean(axis=0), color=color, linewidth=1.5,
                   marker=marker, markersize=5, label=label)

loss_axis.text(-0.05, 56, "69.20", color=blue, fontsize=8)
loss_axis.text(-0.05, 82, "72.32", color=orange, fontsize=8)
loss_axis.text(0.89, 8, "0.72", color=blue, fontsize=8)
loss_axis.text(0.87, 81, "69.74", color=orange, fontsize=8)
loss_axis.set(xlim=(-0.15, 1.15), ylim=(0, 90), yticks=[0, 25, 50, 75])
loss_axis.set_xticks([0, 1], ["100-class\nincoming loss", "Current-ten\nincoming loss"],
                    fontsize=8)
loss_axis.set_ylabel("New accuracy, all classes (%)", fontsize=8)
loss_axis.grid(axis="y", color="#dddddd", linewidth=0.5)
loss_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                 frameon=False, fontsize=8)

# C. Original and adjusted scores are evaluated on the same 8,000 images.
fig.text(0.04, 0.415, "C   Old-only replay, before and after score adjustment",
         weight="bold", fontsize=8.8)
fig.text(0.04, 0.393, "Filled points follow old targets of 5, 10, 15, 20% (+3 pp in validation).",
         fontsize=7.5)
tradeoff_axis = fig.add_axes([0.135, 0.075, 0.825, 0.285])
profile = {}
for name, label, color, marker, linestyle in [
    ("usual_er", "100-class loss", blue, "o", "-"),
    ("current_mask", "Current-ten loss", orange, "^", "--"),
    ("old_blend05", "5% all-class term", green, "s", ":"),
]:
    raw_values = np.array([
        [float(raw_lookup[name, seed][key]) for key in ("old_pct", "new_pct")]
        for seed in seeds
    ])
    adjusted_values = np.array([
        [
            [float(adjusted_lookup[name, seed, target][key])
             for key in ("test_old_pct", "test_new_pct")]
            for target in targets
        ]
        for seed in seeds
    ])
    assert all(raw_lookup[name, seed]["old_n"] == "7200" for seed in seeds)
    assert all(raw_lookup[name, seed]["new_n"] == "800" for seed in seeds)
    assert all(adjusted_lookup[name, seed, target]["available"] == "True"
               for seed in seeds for target in targets)
    assert np.isfinite(raw_values).all() and np.isfinite(adjusted_values).all()

    light = np.array(to_rgb(color)) * 0.25 + 0.75
    for row in adjusted_values:
        tradeoff_axis.plot(row[:, 0], row[:, 1], color=light, linewidth=0.55,
                           linestyle=linestyle, marker=marker, markersize=2.1,
                           markeredgewidth=0.3)
    tradeoff_axis.scatter(raw_values[:, 0], raw_values[:, 1], s=13,
                          facecolor="white", edgecolor=light, marker=marker,
                          linewidths=0.7, zorder=3)
    adjusted_mean = adjusted_values.mean(axis=0)
    raw_mean = raw_values.mean(axis=0)
    tradeoff_axis.plot(adjusted_mean[:, 0], adjusted_mean[:, 1], color=color,
                       linewidth=1.35, linestyle=linestyle, marker=marker,
                       markersize=4.6, label=label, zorder=5)
    tradeoff_axis.scatter(*raw_mean, s=65, facecolor="white", edgecolor=color,
                          marker=marker, linewidths=1.25, zorder=6)
    profile[name] = {
        "raw_old_new_per_seed": raw_values.tolist(),
        "adjusted_old_new_per_seed_target": adjusted_values.tolist(),
        "raw_old_new_mean": raw_mean.tolist(),
        "adjusted_old_new_mean": adjusted_mean.tolist(),
    }

tradeoff_axis.set(xlim=(0, 30), ylim=(-3, 86), xticks=[0, 5, 10, 15, 20, 25, 30],
                  yticks=[0, 20, 40, 60, 80])
tradeoff_axis.set_xlabel("Actual old-class accuracy (%)", fontsize=8.5, labelpad=4)
tradeoff_axis.set_ylabel("New-class accuracy (%)", fontsize=8.5)
tradeoff_axis.tick_params(labelsize=8)
tradeoff_axis.grid(color="#e3e3e3", linewidth=0.5)
tradeoff_axis.legend(loc="lower left", bbox_to_anchor=(0.01, 0.025),
                     frameon=False, fontsize=7.6, handlelength=2.5,
                     borderaxespad=0, labelspacing=0.4)
tradeoff_axis.text(1.6, 39, "Large open markers\nshow original scores",
                   ha="left", va="center", fontsize=7.5)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
clipped_text = []
for text_item in fig.findobj(matplotlib.text.Text):
    if text_item.get_visible() and text_item.get_text():
        extent = text_item.get_window_extent(renderer)
        if (extent.x0 < 0 or extent.y0 < 0 or extent.x1 > fig.bbox.width
                or extent.y1 > fig.bbox.height):
            clipped_text.append(text_item.get_text())
assert not clipped_text, clipped_text

for extension in ["pdf", "svg", "eps", "png"]:
    fig.savefig(OUTPUT / f"overview.{extension}", dpi=300, facecolor="white")
plt.close(fig)
with Image.open(OUTPUT / "overview.png") as preview:
    preview.convert("L").save(OUTPUT / "overview_gray.png")
(OUTPUT / "overview_data_profile.json").write_text(json.dumps({
    "paired_seeds": seeds,
    "task": 9,
    "panel_b_test_images": 10000,
    "panel_c_test_images": 8000,
    "panel_c_validation_images": 2000,
    "panel_c_old_targets_pct": targets,
    "panel_c_validation_margin_pp": 3,
    "held_out_scores": profile,
    "clipped_text": clipped_text,
}, indent=2))
print("Overview plotted from five paired histories and identical held-out images.")
