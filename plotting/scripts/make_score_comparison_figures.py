"""Draw the score-shift illustration and summarize paired new-accuracy effects."""
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from statistics import mean
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifact"
if not ARTIFACT.is_dir():
    ARTIFACT = ROOT.parent
EVIDENCE = ARTIFACT / "evidence"
sys.path.insert(0, str(ARTIFACT / "tools"))
from compare_old_accuracy import at_old_target, correct, read_original

OLD = "#244f73"
NEW = "#a64e2e"
INK = "#26333e"


def save(fig, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def score_shift():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75), sharey=True)
    x = np.linspace(-3.5, 1.0, 501)
    arrangements = (
        ("a  Distinct scores", -2.45, -0.79, .29, -1.62),
        ("b  Overlapping scores", -1.42, -.66, .59, -1.02),
    )
    for ax, (title, old, new, width, shifted) in zip(axes, arrangements):
        for center, color, style, label in (
            (old, OLD, "-", "Old images"), (new, NEW, "--", "New images")
        ):
            height = np.exp(-.5*((x-center)/width)**2)
            ax.plot(x, height, color=color, lw=1.7, ls=style, label=label)
            ax.fill_between(x, height, color=color, alpha=.11)
        ax.vlines(0, 0, 1.10, color="#8a8b8d", lw=1.0, linestyles=(0, (2, 3)))
        ax.vlines(shifted, 0, 1.10, color=INK, lw=1.0)
        ax.annotate("", xy=(shifted, 1.16), xytext=(0, 1.16),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
        ax.text(shifted/2, 1.23, "Move cutoff",
                ha="center", va="bottom", fontsize=9.5)
        ax.set_title(title, fontsize=11, loc="left", weight="bold")
        ax.set_xticks([shifted, 0], ["Shifted\ncutoff", "Original\ncutoff"])
        ax.set(xlim=(-3.5, 1), ylim=(0, 1.43))
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4, labelsize=9.7)
    fig.supxlabel("Highest new score minus highest old score", fontsize=9.5, y=.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .99), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.03, right=.985, bottom=.24, top=.77, wspace=.22)
    save(fig, "score_shift_concept")


def held_out_effect():
    """Compare the same five task-ten pairs on the same held-out images."""
    raw, matched = [], []
    for seed in range(3101, 3106):
        pair = []
        for condition in ("restricted", "full"):
            record = read_original(EVIDENCE, 9, seed, condition)
            labels = record["mapped_labels"]
            test = ~record["calibration"]
            old_n = int((test & (labels < 90)).sum())
            new_n = int(test.sum()) - old_n
            margin, old_ok, new_ok = correct(record, labels, 90)
            old_hits, new_hits = at_old_target(margin, test & old_ok, test & new_ok, old_n, 20)
            assert 100 * old_hits == 20 * old_n
            pair.append((100 * int((test & new_ok & (margin < 0)).sum())/new_n,
                         100 * new_hits/new_n))
        raw.append(pair[0][0] - pair[1][0])
        matched.append(pair[0][1] - pair[1][1])
    return mean(raw), mean(matched)


def saved_effect(family, filename, step, policy, reference=None):
    source = json.loads((EVIDENCE / family / filename).read_text(encoding="utf-8"))
    summaries = (source["summaries"][step], source["measured_old20"]["steps"][step])
    output = []
    for results in summaries:
        effects = np.asarray(results[policy]["paired"]["new"]["seeds"])
        if reference is not None:
            effects = effects - np.asarray(results[reference]["paired"]["new"]["seeds"])
        output.append(float(effects.mean()))
    return tuple(output)


def key_effects():
    c10 = "reservoir_supervision_timing"
    c100 = "er_ace_timing_c100"
    rows = [
        ("Ten-class vs full loss\nCIFAR-100, task 10", held_out_effect()),
        ("Omission vs reversal\nCIFAR-10, single pass",
         saved_effect(c10, "order_omission_summary.json", "157", "omit", "decreasing")),
        ("Omission vs reversal\nCIFAR-100, random order",
         saved_effect(c100, "omission_summary.json", "250", "omit", "decreasing")),
        ("Exchange vs reversal\nTask end, 95 cases",
         saved_effect(c100, "exchange_summary.json", "250", "exchanged")),
        ("Exchange vs reversal\n100 updates, 15 cases",
         saved_effect(c100, "continuation_summary.json", "100", "exchanged")),
    ]
    fig = plt.figure(figsize=(6.35, 3.95))
    cells = fig.add_gridspec(1, 3, width_ratios=[1.9, 3.0, 1.48], wspace=.045)
    labels, ax, numbers = [fig.add_subplot(cells[0, i]) for i in range(3)]
    for x in (labels, ax, numbers):
        x.set_ylim(-.6, 4.6)
        x.set_yticks([])
    for x in (labels, numbers):
        x.set_xlim(0, 1)
        x.axis("off")
    for line in (.5, 1.5, 2.5, 3.5):
        ax.axhline(line, color=".9", lw=.65, zorder=0)
    ax.axvline(0, color=".54", lw=.75, ls=":", zorder=0)
    for y, (label, (raw, matched)) in zip(range(4, -1, -1), rows):
        labels.text(.015, y, label, color=INK, fontsize=10, va="center", linespacing=1.35)
        ax.plot([raw, matched], [y+.105, y-.105], lw=1.4, color=".57", zorder=1)
        ax.scatter(raw, y+.105, s=43, marker="o", facecolors="white",
                   edgecolors=INK, linewidths=1.2, zorder=2)
        ax.scatter(matched, y-.105, s=40, marker="D", color=OLD, zorder=3)
        for x, value, color in ((.32, raw, INK), (.88, matched, OLD)):
            rounded = Decimal(f"{value:.8f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            numbers.text(x, y, f"{rounded:+.2f}", fontsize=10.2, ha="right", va="center", color=color)
    numbers.text(.32, 1.02, "Original", transform=numbers.transAxes, fontsize=10, ha="right")
    numbers.text(.88, 1.02, "At 20%", transform=numbers.transAxes,
                 fontsize=10, ha="right", color=OLD)
    ax.set_xlim(-73, 23)
    ax.set_xticks([-70, -50, -30, -10, 0, 20])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("New-class accuracy difference (pp)", fontsize=10)
    ax.spines[["top", "left", "right"]].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor=INK, label="Original scores"),
        plt.Line2D([], [], marker="D", ls="", color=OLD, label="At 20% old accuracy"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .99), frameon=False,
               ncol=2, fontsize=10)
    fig.subplots_adjust(left=.02, right=.994, bottom=.16, top=.85)
    save(fig, "key_effects")
    for label, (raw, matched) in rows:
        print(label.replace("\n", ", "), f"{raw:+.4f}", f"{matched:+.4f}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    score_shift()
    key_effects()
