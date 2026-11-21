"""Summarize paired training trajectories from recorded scores and exposure."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if ROOT.name != "artifact" and not (ROOT / "tools/compare_old_accuracy.py").is_file():
    ROOT = ROOT / "artifact"
sys.path.insert(0, str(ROOT / "tools"))
from compare_old_accuracy import at_old_target, load, stats


def metrics(data, q):
    y = data["labels"]
    new = np.isin(y, data["new_classes"])
    old = ~new
    old_ok = old & (data["old_prediction"] == y)
    new_ok = new & (data["new_prediction"] == y)
    old_n = int(old.sum())
    old_ceiling = 100*old_ok.sum()/old_n
    matched = None
    if old_ceiling >= q:
        margin = data["old_score"].astype(float) - data["new_score"].astype(float)
        kept, correct = at_old_target(margin, old_ok, new_ok, old_n, q)
        matched = dict(old=100*kept/old_n, new=100*correct/int(new.sum()))
    return dict(overall=100*float((data["predicted"] == y).mean()),
                new=100*float((data["predicted"][new] == y[new]).mean()),
                old=100*float((data["predicted"][old] == y[old]).mean()),
                within_new=100*float(new_ok.sum()/new.sum()), within_old=float(old_ceiling),
                fixed_old=matched)


def records_in(data_dir, reference_policy, extra_reference=None):
    training = json.loads((data_dir/"training.json").read_text(encoding="utf-8"))
    for row in training:
        yield data_dir, row
    if extra_reference is not None:
        training = json.loads((extra_reference/"training.json").read_text(encoding="utf-8"))
        for row in training:
            if row["policy"] == reference_policy:
                yield extra_reference, row


def compare(data_dir, target, reference_policy="retrieval", extra_reference=None):
    indexed = {}
    for source, row in records_in(data_dir, reference_policy, extra_reference):
        key = row["seed"], row["task"], row["policy"]
        directory = source / f"s{key[0]}_t{key[1]}_{key[2]}"
        indexed[key] = {}
        for step, dose in row["dose_by_update"].items():
            step = int(step)
            values = metrics(load(directory / f"{step}.npz"), target)
            indexed[key][step] = dict(values, dose=dose)
    seeds = sorted({key[0] for key in indexed})
    tasks = sorted({key[1] for key in indexed})
    policies = sorted({key[2] for key in indexed})
    steps = sorted(indexed[seeds[0],tasks[0],policies[0]])
    scalar_metrics = ("new", "old", "overall", "within_new", "within_old", "dose")

    summaries = {}
    for step in steps:
        summary = {}
        for policy in policies:
            paired = {}
            absolute = {}
            by_task = {}
            for field in scalar_metrics:
                means = []
                differences = []
                for seed in seeds:
                    task_values = [indexed[seed,task,policy][step][field] for task in tasks]
                    reference = [indexed[seed,task,reference_policy][step][field] for task in tasks]
                    means.append(mean(task_values))
                    differences.append(mean(a-b for a,b in zip(task_values,reference)))
                paired[field] = stats(differences)
                absolute[field] = stats(means)
            pairs = []
            for task in tasks:
                deltas = {}
                for field in ("new", "old"):
                    deltas[field] = stats([indexed[seed,task,policy][step][field] -
                                           indexed[seed,task,reference_policy][step][field] for seed in seeds])
                by_task[task] = deltas
                for seed in seeds:
                    first = indexed[seed,task,policy][step]["fixed_old"]
                    reference = indexed[seed,task,reference_policy][step]["fixed_old"]
                    if first is not None and reference is not None:
                        pairs.append({"seed":seed,"task":task,"new_difference":first["new"]-reference["new"],
                                      "old":first["old"],"reference_old":reference["old"]})
            summary[policy] = dict(means=absolute, paired=paired, by_task=by_task, matched=pairs)
        summaries[step] = summary
    return dict(states=len(indexed), seeds=seeds, tasks=tasks, matched_old_target=target,
                summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-target", type=int, default=20)
    parser.add_argument("--reference", default="retrieval")
    args = parser.parse_args()
    report = compare(args.data, args.old_target, args.reference)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for step, policies in report["summaries"].items():
        print(step, {policy: {field: round(row["paired"][field]["mean"], 4)
                             for field in ("new", "old", "within_old", "dose")}
                     for policy, row in policies.items()})


if __name__ == "__main__":
    main()
