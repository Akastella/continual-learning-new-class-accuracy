"""Plot training progress for identical-image and expected-count placements."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "new_experiments/reservoir_supervision_timing/summary.json"
data = json.loads(source.read_text(encoding="utf-8"))["summaries"]
steps = [int(step) for step in data]
policies = [
    ("retrieval", "At retrieval", "#005B9A", "-", "o"),
    ("first_presentation", "At first presentation", "#AB4D18", "--", "s"),
    ("expected_count", "Expected count", "#087A69", ":", "^"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.3, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.50), sharex=True)
for label, field, ax in zip(("a  Loss weight", "b  New accuracy", "c  Old accuracy"),
                            ("dose", "new", "old"), axes):
    for policy, description, color, style, marker in policies:
        values = [data[str(step)][policy]["means"][field]["mean"] for step in steps]
        if field == "dose":
            values = [100*v/values[-1] for v in values]
        ax.plot(steps, values, label=description, color=color, linestyle=style, marker=marker,
                linewidth=1.6, markersize=4.4, markerfacecolor="white")
    ax.set_title(label, loc="left", weight="bold")
    ax.set_xticks([0, 40, 100, 157])
    ax.set_xlim(0, 163)
    ax.grid(axis="y", linewidth=.55, alpha=.33)
    ax.set_axisbelow(True)
axes[0].set_ylim(-4, 106)
axes[1].set_ylim(-.8, 22)
axes[2].set_ylim(37, 48)
fig.supxlabel("Updates since task began", y=.17, size=9.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(.5, .005), handlelength=2.5)
fig.subplots_adjust(left=.09, right=.965, bottom=.35, top=.90, wspace=.39)
for suffix in ("pdf", "svg", "png"):
    fig.savefig(ROOT / "figures" / f"reservoir_timing.{suffix}", dpi=205)
