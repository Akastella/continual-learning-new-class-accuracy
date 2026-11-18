"""Build the Fashion-MNIST placement comparison table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "new_experiments/fashion_placement/summary.json"
values = json.loads(source.read_text(encoding="utf-8"))["summaries"]["188"]
conditions = [
    ("retrieval", "At retrieval"),
    ("same_step", "Scheduled images, same step"),
    ("expected_count", "Scheduled images, expected count"),
    ("first_presentation", "At first presentation"),
    ("omitted", "Omitted"),
]
rows = [r"\begin{tabular}{lrrr}", r"\toprule",
        r"Placement & New & Within new & Old\\", r"\midrule"]
for policy, label in conditions:
    scores = values[policy]["means"]
    cells = [f"${scores[k]['mean']:.2f}\\pm{scores[k]['sample_sd']:.2f}$"
             for k in ("new", "within_new", "old")]
    rows.append(" & ".join([label, *cells]) + r"\\")
rows.extend([r"\bottomrule", r"\end{tabular}"])
(ROOT / "tables/fashion_placement.tex").write_text("\n".join(rows)+"\n", encoding="utf-8")
