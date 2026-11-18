"""Generate endpoint table for the twenty-task CIFAR-100 timing comparison."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "new_experiments/er_ace_timing_c100"
LABELS = {
    "retrieval":"At retrieval",
    "same_step":"Scheduled images, same step",
    "expected_count":"Scheduled images, expected count",
    "first_presentation":"At first presentation",
    "half_at_first":"Half first, half at retrieval",
}

def write_table(filename, last_task=False):
    rows = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Group supervision & New & Old & Overall\\", r"\midrule"]
    for name, label in (("summary.json", "10,000 images"),
                        ("smaller_summary.json", "2,000 images")):
        data = json.loads((SOURCE/name).read_text(encoding="utf-8"))
        result = data["last_task"] if last_task else data["summaries"]["250"]
        rows.append(r"\multicolumn{4}{l}{Memory of " + label + r"}\\")
        for key, description in LABELS.items():
            if key in result:
                values = result[key]["means"]
                cells = ["$"+f"{values[k]['mean']:.2f}\\pm{values[k]['sample_sd']:.2f}"+"$"
                         for k in ("new", "old", "overall")]
                rows.append(" & ".join([description, *cells]) + r"\\")
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    (ROOT/"tables"/filename).write_text("\n".join(rows)+"\n", encoding="utf-8")


write_table("er_ace_timing_c100.tex")
write_table("er_ace_timing_c100_last.tex", last_task=True)
