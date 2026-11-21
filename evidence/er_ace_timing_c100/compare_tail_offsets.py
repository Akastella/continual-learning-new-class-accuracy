"""Match another schedule's validation retention with one score offset."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"reservoir_supervision_timing"))
from calibrate_timing import validation_split
from recompute import load, stats


def predict(record, offset):
    margin = record["old_score"].astype(float)-record["new_score"].astype(float)
    return np.where(margin >= offset, record["old_prediction"], record["new_prediction"])


def evaluate(predicted, labels, new, subset):
    correct = predicted == labels
    return dict(old=100*float(correct[subset & ~new].mean()),
                new=100*float(correct[subset & new].mean()),
                overall=100*float(correct[subset].mean()))


def compare(source):
    records = json.loads((source/"training.json").read_text(encoding="utf-8"))
    seeds = sorted({r["seed"] for r in records})
    tasks = sorted({r["task"] for r in records})
    rows = {policy: [] for policy in ("large_increasing", "large_decreasing")}
    for seed in seeds:
        for task in tasks:
            baseline = load(source/f"s{seed}_t{task}_decreasing"/"250.npz")
            labels = baseline["labels"]
            old = ~np.isin(labels, baseline["new_classes"])
            val = validation_split(labels)
            eligible = val & old & (baseline["old_prediction"] == labels)
            margin = baseline["old_score"].astype(float)-baseline["new_score"].astype(float)
            cutoffs = np.sort(margin[eligible])
            for policy, results in rows.items():
                trained = load(source/f"s{seed}_t{task}_{policy}"/"250.npz")
                assert np.array_equal(labels, trained["labels"])
                assert np.array_equal(baseline["new_classes"], trained["new_classes"])
                required = int(((trained["predicted"] == labels) & val & old).sum())
                if required > len(cutoffs):
                    continue
                offset = cutoffs[-required] if required else np.nextafter(margin[val].max(), np.inf)
                original = baseline["predicted"]
                adjusted = predict(baseline, offset)
                target = trained["predicted"]
                test = ~val
                results.append(dict(seed=seed, task=task, offset=float(offset),
                                    validation_target=100*required/int((val & old).sum()),
                                    baseline_validation=evaluate(adjusted, labels, ~old, val),
                                    baseline_test=evaluate(original, labels, ~old, test),
                                    fitted=evaluate(adjusted, labels, ~old, test),
                                    trained=evaluate(target, labels, ~old, test),
                                    agreement_before=100*float((original[test]==target[test]).mean()),
                                    agreement_after=100*float((adjusted[test]==target[test]).mean())))
    report = {}
    for policy, results in rows.items():
        differences = {}
        for measure in ("new", "old", "overall"):
            means = [mean(r["fitted"][measure]-r["trained"][measure]
                          for r in results if r["seed"] == seed)
                     for seed in sorted({r["seed"] for r in results})]
            differences[measure] = stats(means) if len(means)>1 else None
        agreement = {
            name: stats([mean(r[name] for r in results if r["seed"] == seed)
                         for seed in sorted({r["seed"] for r in results})])
            for name in ("agreement_before", "agreement_after")
        }
        report[policy] = dict(total=len(seeds)*len(tasks), feasible=len(results),
                              differences=differences, agreement=agreement, pairs=results)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.data)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for policy, comparison in result.items():
        print(policy, comparison["feasible"], comparison["differences"])


if __name__ == "__main__":
    main()
