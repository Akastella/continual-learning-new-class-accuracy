"""Compare observed old accuracy on original and replacement images."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def correct(record, labels, old):
    margin = record["old_score"].astype(float) - record["current_score"].astype(float)
    current_key = "current_prediction" if "current_prediction" in record else "predictions_current10"
    return (margin, (labels < old) & (record["old_prediction"] == labels),
            (labels >= old) & (record[current_key] == labels))


def stats(values):
    return {"mean": mean(values), "sample_sd": stdev(values), "seeds": values}


def group_auroc(record, test, old):
    labels = record["mapped_labels"]
    margin = record["old_score"].astype(float) - record["current_score"].astype(float)
    old_scores = np.sort(margin[test & (labels < old)])
    new_scores = margin[test & (labels >= old)]
    left = np.searchsorted(old_scores, new_scores, side="left")
    right = np.searchsorted(old_scores, new_scores, side="right")
    return float(100 * (1 - ((left + right) / (2 * len(old_scores))).mean()))


def read_original(evidence, task, seed, condition):
    if task == 7:
        folder = evidence / "supervision_task7/reader_data"
        method = "usual_er" if condition == "full" else "old_restricted"
    else:
        folder = evidence / "cross_group_supervision/score_analysis/inputs"
        method = "usual_er" if condition == "full" else "current_mask"
    data = load(folder / f"s{seed}_{method}.npz")
    rows = data["official_test_indices"]
    assert np.array_equal(rows, np.arange(len(rows)))
    if "calibration" not in data:
        calib = np.zeros(len(rows), dtype=bool)
        for label in range(100):
            members = rows[data["original_labels"] == label]
            rng = np.random.Generator(np.random.PCG64(2026110800 + label))
            calib[rng.permutation(members)[:20]] = True
        data["calibration"] = calib
    return data


def compare(evidence):
    records = []
    for task in (7, 9):
        old, seen = task * 10, (task + 1) * 10
        for condition in ("full", "restricted"):
            for seed in range(3101, 3106):
                x = read_original(evidence, task, seed, condition)
                y = load(evidence / "cifair_replacements/reader_data" / f"t{task}_s{seed}_{condition}.npz")
                rows, offset = y["test_rows"], float(y["offset"])
                assert np.array_equal(x["mapped_labels"][rows], y["mapped_labels"])
                assert np.array_equal(x["original_labels"][rows], y["original_labels"])
                match = ~x["calibration"][rows] & (y["mapped_labels"] < old)
                old_labels = y["mapped_labels"] < old
                old_n = int(old_labels.sum())
                mx, ox, nx = correct(x, x["mapped_labels"], old)
                my, oy, _ = correct(y, y["mapped_labels"], old)
                matched_n = int(match.sum())
                first = int((ox[rows][match] & (mx[rows][match] >= offset)).sum())
                second = int((oy[match] & (my[match] >= offset)).sum())
                cap = int(oy.sum())
                actual = int((oy & (my >= offset)).sum())

                # Trace a test-set operating point at the same measured old accuracy.
                test = ~x["calibration"] & (x["mapped_labels"] < seen)
                test_old_n = int((test & (x["mapped_labels"] < old)).sum())
                test_new_n = int(test.sum()) - test_old_n
                eligible = np.sort(mx[test & ox])
                needed = (20 * test_old_n + 99) // 100
                assert len(eligible) >= needed
                threshold = eligible[-needed]
                fixed_old_correct = int((test & ox & (mx >= threshold)).sum())
                fixed_new_correct = int((test & nx & (mx < threshold)).sum())
                records.append(dict(task=task + 1, seed=seed, condition=condition,
                    old_n=old_n, old_cap_correct=cap, old_correct=actual,
                    matched_n=matched_n, original_matched_correct=first,
                    replacement_matched_correct=second, fixed_old_correct=fixed_old_correct,
                    fixed_new_correct=fixed_new_correct, fixed_old_n=test_old_n, fixed_new_n=test_new_n,
                    group_auc=group_auroc(x, test, old)))
    return records


def summarize(records):
    results = {}
    for task in (8, 10):
        rows = [r for r in records if r["task"] == task]
        result = {}
        for condition in ("full", "restricted"):
            group = [r for r in rows if r["condition"] == condition]
            result[condition] = dict(
                original_matched_old=stats([100*r["original_matched_correct"]/r["matched_n"] for r in group]),
                replacement_matched_old=stats([100*r["replacement_matched_correct"]/r["matched_n"] for r in group]),
                old_cap_below20=sum(5*r["old_cap_correct"] < r["old_n"] for r in group),
                old_adjusted_below20=sum(5*r["old_correct"] < r["old_n"] for r in group),
                fixed_old_actual=stats([100*r["fixed_old_correct"]/r["fixed_old_n"] for r in group]),
            )
        full, restricted = ([r for r in rows if r["condition"] == c] for c in ("full", "restricted"))
        result["fixed_old_new_difference"] = stats([
            100*(b["fixed_new_correct"]-a["fixed_new_correct"])/a["fixed_new_n"]
            for a,b in zip(full,restricted)
        ])
        result["group_auc_difference"] = stats([
            b["group_auc"]-a["group_auc"] for a,b in zip(full,restricted)
        ])
        results[task] = result
    return {"results": results, "individual_counts": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.artifact / "evidence"
    result = summarize(compare(evidence))
    for task in (7, 9):
        changes = []
        for seed in range(3101, 3106):
            full = read_original(evidence, task, seed, "full")
            test = ~full["calibration"] & (full["mapped_labels"] < (task + 1) * 10)
            auc = []
            for method in ("mixed_restricted", "revisit_current10"):
                data = load(evidence / "revisit_supervision/reader_data" / f"t{task}_s{seed}_{method}.npz")
                assert np.array_equal(data["mapped_labels"], full["mapped_labels"])
                auc.append(group_auroc(data, test, task * 10))
            changes.append(auc[1]-auc[0])
        result["results"][task+1]["mixed_group_auc_difference"] = stats(changes)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for task, values in result["results"].items():
        print(task, "fixed-old new difference", values["fixed_old_new_difference"]["mean"],
              "matched old accuracy", [(round(values[c]["original_matched_old"]["mean"], 2),
                  round(values[c]["replacement_matched_old"]["mean"], 2)) for c in ("full","restricted")])


if __name__ == "__main__":
    main()
