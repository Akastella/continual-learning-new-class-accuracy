"""Show the weight schedules, their task-end effect, and its persistence."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "new_experiments/er_ace_timing_c100"
if not (SOURCE/"order_measurements/training.json").is_file():
    SOURCE = ROOT.parent / "evidence/er_ace_timing_c100"


def read(name):
    return json.loads((SOURCE/name).read_text(encoding="utf-8"))


def build():
    records = read("order_measurements/training.json") + read("exchange_run/training.json")
    exchange = read("exchange_summary.json")
    continuation = read("continuation_summary.json")

    increasing = ("increasing", "Increasing", "#005B9A", "-", "o")
    descending = ("decreasing", "Descending", "#5D6067", ":", "s")
    exchanged = ("exchanged", "Window exchange", "#AB4D18", "--", "^")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(5.55, 4.24))
    cells = fig.add_gridspec(2, 2, height_ratios=(1, .92))
    schedules = fig.add_subplot(cells[0, 0])
    endpoint = fig.add_subplot(cells[0, 1])
    later = fig.add_subplot(cells[1, :])

    for start, end in ((.5, 63.5), (187.5, 250.5)):
        schedules.axvspan(start, end, color=".92", zorder=0)
    for policy, label, color, style, _ in (increasing, descending, exchanged):
        task = next(row for row in records if row["task"] == 10 and row["policy"] == policy)
        schedules.plot(range(1, 251), [100*x for x in task["coefficients"]],
                       color=color, ls=style, lw=1.6, label=label)
    schedules.set_title("a  Same weights, different order", loc="left", fontsize=9, weight="bold")
    schedules.set(xlabel="Update", ylabel="Loss weight (%)", xlim=(0, 251))
    schedules.set_xticks([1, 63, 125, 188, 250])

    scores = (exchange["summaries"]["250"], exchange["measured_old20"]["steps"]["250"])
    endpoint.axhline(0, color=".45", lw=.7)
    for index, metric in enumerate(scores):
        for offset, (policy, _, color, _, marker) in zip((-.12, .12), (increasing, exchanged)):
            value = metric[policy]["paired"]["new"]
            x = index+offset
            endpoint.scatter([x]*len(value["seeds"]), value["seeds"], color=color, marker=marker,
                             alpha=.65, s=16, zorder=3)
            endpoint.hlines(value["mean"], x-.08, x+.08, color=color, lw=1.6)
    endpoint.set_title("b  Task-end new accuracy", loc="left", fontsize=9, weight="bold")
    endpoint.set(ylabel="Gain over descending (pp)", xlim=(-.33, 1.33), ylim=(-2, 24))
    endpoint.set_xticks([0, 1], ["Original\nscores", "20% old\naccuracy"])

    steps = (0, 25, 100, 250)
    later.axhline(0, color=".45", lw=.7)
    for label, values, color, style, marker in (
        ("Original scores", continuation["summaries"], exchanged[2], "-", "^"),
        ("At 20% old accuracy", continuation["measured_old20"]["steps"], "#276C68", "--", "o"),
    ):
        paired = [values[str(step)]["exchanged"]["paired"]["new"] for step in steps]
        later.errorbar(steps, [v["mean"] for v in paired],
                       yerr=[v["sample_sd"] for v in paired], color=color, ls=style,
                       marker=marker, lw=1.5, ms=4, capsize=2, label=label)
    later.set_title("c  The gain after further training", loc="left", fontsize=9, weight="bold")
    later.set(xlabel="Updates on the next task", ylabel="Gain from exchange (pp)",
              xlim=(-10, 260), ylim=(-3, 23))
    later.set_xticks(steps)
    later.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.4, fontsize=8)
    for axis in (schedules, endpoint, later):
        axis.grid(axis="y", alpha=.28, lw=.5)
        axis.set_axisbelow(True)
    fig.legend(*schedules.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.55, .995), columnspacing=1.55)
    fig.subplots_adjust(top=.835, bottom=.139, left=.135, right=.982, wspace=.62, hspace=.91)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT/"figures"/f"c100_order.{suffix}", dpi=205)
    plt.close(fig)


if __name__ == "__main__":
    build()
