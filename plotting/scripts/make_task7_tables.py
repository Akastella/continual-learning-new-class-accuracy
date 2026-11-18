"""Generate earlier-boundary tables from the recorded five-seed summary."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "new_experiments/supervision_task7/summary.json").read_text())
labels = {
    "usual_er": "Old only, 80-way CE",
    "old_restricted": "Old only, 10-way CE",
    "old_blend05": "Old only, 95/5 loss",
    "mixed_full": "Mixed, 80-way CE",
    "mixed_restricted": "Mixed, 10-way CE",
    "common_sgd": "Separate current-only run",
}


def accuracy(value):
    return f"${value['mean']:.2f}\\pm{value['sample_sd']:.2f}$"


raw = [r"\begin{tabular}{lrrrr}", r"\toprule",
       r"Replay and incoming loss & \shortstack{New\\80-way} & \shortstack{New\\10-way} & \shortstack{Old\\80-way} & \shortstack{Overall\\80-way}\\",
       r"\midrule"]
score = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Replay and incoming loss & \shortstack{Eligible\\seeds} & Old & New & \shortstack{Test target\\misses}\\", r"\midrule"]
for method, label in labels.items():
    endpoint = summary["methods"][method]
    raw.append(label + " & " + " & ".join(
        accuracy(endpoint[key]) for key in ("new_pct", "current10_pct", "old_pct", "all_pct")
    ) + r"\\")
    selected = next(row for row in summary["adjusted_test"]
                    if row["method"] == method and row["q"] == 20)
    count = selected["available_seeds"]
    fields = ([f"{count}/5", accuracy(selected["old_pct"]), accuracy(selected["new_pct"]),
               f"{len(selected['test_violations'])}/{count}"] if count else ["0/5", "--", "--", "--"])
    score.append(label + " & " + " & ".join(fields) + r"\\")

individual = [r"\begin{tabular}{rrrrr}", r"\toprule",
              r"Seed & \shortstack{New\\interaction} & \shortstack{Competition\\interaction} & \shortstack{New minus\\half ER} & \shortstack{Old minus\\ER}\\",
              r"\midrule"]
for index, seed in enumerate(summary["seeds"]):
    values = [summary["H1"]["new_interaction"]["values"][index],
              summary["H1"]["competition_interaction"]["values"][index],
              summary["H2"]["new_minus_half_er"]["values"][index],
              summary["H2"]["old_minus_er"]["values"][index]]
    individual.append(str(seed) + " & " + " & ".join(f"${value:+.2f}$" for value in values) + r"\\")

for name, rows in [("task7_raw", raw), ("task7_score20", score), ("task7_individual", individual)]:
    (ROOT / "tables" / f"{name}.tex").write_text(
        "\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8"
    )
print("Generated three earlier-boundary tables.")
