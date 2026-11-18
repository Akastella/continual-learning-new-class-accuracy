"""Generate the single-pass result table from the recomputed seed means."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "new_experiments/full_reservoir_placement/summary.json").read_text())["results"]
lines = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Group term on new images & New & Within new & Old\\", r"\midrule"]
for method, label in (
    ("replay", "Retrieved new images"),
    ("incoming", "Scheduled new images"),
    ("omit", "Omitted"),
):
    values = [data[method][key] for key in ("new", "within_new", "old")]
    cells = ["$" + f"{row['mean']:.2f}\\pm{row['sample_sd']:.2f}$" for row in values]
    lines.append(" & ".join([label, *cells]) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "tables/single_pass.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
