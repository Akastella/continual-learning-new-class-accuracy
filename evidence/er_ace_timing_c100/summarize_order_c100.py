"""Compare group-loss weight schedules on common CIFAR-100 tasks."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"reservoir_supervision_timing"))
from calibrate_timing import apply
from frontiers import frontier
from recompute import compare, load, records_in, stats


def group_scores(data):
    new = np.isin(data["labels"], data["new_classes"])
    margins = data["new_score"].astype(float)-data["old_score"].astype(float)
    old_margins = np.sort(margins[~new])
    new_margins = margins[new]
    left = np.searchsorted(old_margins, new_margins, side="left")
    right = np.searchsorted(old_margins, new_margins, side="right")
    return dict(old_margin=float(old_margins.mean()), new_margin=float(new_margins.mean()),
                auc=100*float((left+right).mean())/(2*len(old_margins)))


def compare_points(source, result, reference, fitted=False):
    policies = result["summaries"][max(result["summaries"])]
    steps = sorted(result["summaries"])
    keys = [(seed, task) for seed in result["seeds"] for task in result["tasks"]]
    common = [key for key in keys if all(source[*key, policy, step] is not None
                                        for policy in policies for step in steps)]
    baseline = "increasing" if reference == "increasing" else "reference"
    measured = {}
    for step in steps:
        comparisons = {}
        for policy in policies:
            pairs = [
                dict(seed=seed, task=task, comparison=source[seed, task, policy, step],
                     **{baseline: source[seed, task, reference, step]})
                for seed, task in common
            ]
            paired = {}
            if len({row["seed"] for row in pairs}) >= 2:
                for field in ("new", "old"):
                    within_seed = [
                        [row["comparison"][field] - row[baseline][field]
                         for row in pairs if row["seed"] == seed]
                        for seed in result["seeds"]
                    ]
                    paired[field] = stats([mean(values) for values in within_seed if values])
            comparisons[policy] = dict(paired=paired, pairs=pairs)
            if fitted:
                comparisons[policy]["misses"] = [
                    sum(row[key]["missed"] for row in pairs) for key in (baseline, "comparison")
                ]
        measured[step] = comparisons
    return dict(total=len(keys), common=len(common),
                by_task={task: sum(t == task for _, t in common) for task in result["tasks"]},
                steps=measured)


def summarize(root, reference="increasing", targets=(), extra_reference=None):
    result = compare(root, 20, reference, extra_reference)
    targets = sorted({20, *targets})
    points = {q: {} for q in targets}
    fitted = {}
    endpoint = {}
    for source, row in records_in(root, reference, extra_reference):
        seed, task, policy = (row[key] for key in ("seed", "task", "policy"))
        folder = source/f"s{seed}_t{task}_{policy}"
        for step in result["summaries"]:
            path = folder/f"{step}.npz"
            data = load(path)
            curve = frontier(data)
            if step == max(result["summaries"]):
                endpoint[seed, task, policy] = group_scores(data)
            for q in targets:
                old, new = curve[q]
                points[q][seed, task, policy, step] = None if new != new else dict(old=old, new=new)
            fitted[seed, task, policy, step] = apply(path, 20)
    for q in targets:
        result[f"measured_old{q}"] = compare_points(points[q], result, reference)
    result["validation_fitted_old20"] = compare_points(fitted, result, reference, fitted=True)
    result["group_score_changes"] = {
        policy: {
            field: stats([
                mean(endpoint[seed, task, policy][field]-endpoint[seed, task, reference][field]
                     for task in result["tasks"])
                for seed in result["seeds"]
            ])
            for field in ("old_margin", "new_margin", "auc")
        }
        for policy in result["summaries"][max(result["summaries"])]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="increasing")
    parser.add_argument("--extra-reference", type=Path)
    parser.add_argument("--additional-old-targets", type=int, nargs="*", default=())
    args = parser.parse_args()
    result = summarize(args.data, args.reference, args.additional_old_targets, args.extra_reference)
    with args.output.open("x", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=2, allow_nan=False)
        outfile.write("\n")
    for step, comparisons in result["summaries"].items():
        for policy, comparison in comparisons.items():
            raw = comparison["paired"]["new"]
            print(step, policy, raw["mean"], raw["sample_sd"])
    for label in ("measured_old20", "validation_fitted_old20"):
        print(label, result[label]["common"])


if __name__ == "__main__":
    main()
