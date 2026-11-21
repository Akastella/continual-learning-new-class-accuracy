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


def ranking_auc(sorted_old, new):
    left = np.searchsorted(sorted_old, new, side="left")
    right = np.searchsorted(sorted_old, new, side="right")
    return 100*float((left+right).mean())/(2*len(sorted_old))


def group_scores(data):
    new = np.isin(data["labels"], data["new_classes"])
    margin = data["new_score"].astype(float)-data["old_score"].astype(float)
    old_margin = np.sort(margin[~new])
    return dict(old_margin=float(old_margin.mean()), new_margin=float(margin[new].mean()),
                auc=ranking_auc(old_margin, margin[new]))


def group_diagnostics(root, result, reference, extra_reference, policies):
    step = max(result["summaries"])
    target = result["matched_old_target"]
    paired = result[f"measured_old{target}"]["steps"][step]
    common = {(r["seed"], r["task"]) for r in paired[reference]["pairs"]}
    by_case = {key: {} for key in common}
    for source, row in records_in(root, reference, extra_reference, policies):
        seed, task, policy = (row[k] for k in ("seed", "task", "policy"))
        if (seed, task) in by_case:
            by_case[seed, task][policy] = load(source/f"s{seed}_t{task}_{policy}"/f"{step}.npz")
    rows, coverage = [], []
    for (seed, task), data in sorted(by_case.items()):
        baseline = data[reference]
        y = baseline["labels"]
        new = np.isin(y, baseline["new_classes"])
        old = ~new
        correct = {}
        for policy, scores in data.items():
            assert np.array_equal(scores["labels"], y)
            assert np.array_equal(scores["new_classes"], baseline["new_classes"])
            correct[policy] = np.where(new, scores["new_prediction"], scores["old_prediction"]) == y
        common_correct = np.logical_and.reduce(list(correct.values()))
        coverage.append(dict(seed=seed, task=task,
                             old=int((old & common_correct).sum()), new=int((new & common_correct).sum())))
        for policy in sorted(data):
            scores = data[policy]
            margin = scores["new_score"].astype(float)-scores["old_score"].astype(float)
            needed = (target*int(old.sum())+99)//100
            cutoff = np.sort(margin[old & correct[policy]])[needed-1]
            eligible_new = new & correct[policy]
            rows.append(dict(
                seed=seed, task=task, policy=policy,
                old_ceiling=100*float(correct[policy][old].mean()),
                new_ceiling=100*float(correct[policy][new].mean()),
                new_group_win=100*float((margin[eligible_new]>cutoff).mean()),
                common_correct_auc=ranking_auc(np.sort(margin[old & common_correct]), margin[new & common_correct])))
    means = {}
    for policy in paired:
        means[policy] = {}
        for metric in ("old_ceiling", "new_ceiling", "new_group_win", "common_correct_auc"):
            by_seed = {seed: mean(r[metric] for r in rows if r["seed"] == seed and r["policy"] == policy)
                       for seed in sorted({r["seed"] for r in rows})}
            means[policy][metric] = dict(absolute=stats(list(by_seed.values())),
                                         paired=stats([value-mean(
                                             r[metric] for r in rows if r["seed"] == seed and r["policy"] == reference)
                                             for seed, value in by_seed.items()]))
    return dict(total=len(common), estimates=means, common_correct_counts=coverage, cases=rows)


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


def summarize(root, reference="increasing", targets=(), extra_reference=None,
              policies=None, original_label_split=False, describe_groups=False):
    result = compare(root, 20, reference, extra_reference, policies)
    targets = sorted({20, *targets})
    points = {q: {} for q in targets}
    fitted = {}
    endpoint = {}
    for source, row in records_in(root, reference, extra_reference, policies):
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
            class_order = row["class_order"] if original_label_split else None
            fitted[seed, task, policy, step] = apply(path, 20, class_order)
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
    if describe_groups:
        result["correct_group_comparison"] = group_diagnostics(root, result, reference, extra_reference, policies)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="increasing")
    parser.add_argument("--extra-reference", type=Path)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--original-label-split", action="store_true")
    parser.add_argument("--describe-groups", action="store_true")
    parser.add_argument("--additional-old-targets", type=int, nargs="*", default=())
    args = parser.parse_args()
    result = summarize(args.data, args.reference, args.additional_old_targets, args.extra_reference,
                       args.policies, args.original_label_split, args.describe_groups)
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
