"""Summarize the paired experiment that relocates all-class supervision."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("mixed_restricted", "revisit_current10", "all_seen_on_nonrepeated")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def stats(values):
    return {"mean": mean(values), "sample_sd": stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = args.artifact / "evidence/revisit_supervision"
    analysis = module("prior_measurement", prior / "recompute.py")
    score = module("group_ranking", args.artifact / "tools/compare_old_accuracy.py")
    summary, records = {}, []
    for task in (7, 9):
        old, seen = task * 10, (task + 1) * 10
        raw_differences, gains, required = [], [], []
        for seed in range(3101, 3106):
            sample_labels = None
            measured = []
            for method in METHODS:
                source = HERE if method == METHODS[-1] else prior
                filename = source / "reader_data" / f"t{task}_s{seed}_{method}.npz"
                with np.load(filename, allow_pickle=False) as archive:
                    data = dict(archive)
                labels = data["mapped_labels"]
                if sample_labels is None:
                    sample_labels = labels
                else:
                    assert np.array_equal(sample_labels, labels)
                seen_images = labels < seen
                calibration = analysis.calibration_mask(data["original_labels"]) & seen_images
                test = seen_images & ~calibration
                raw = analysis.measure(data, seen_images)
                max_old = calibration & (labels < old) & (data["old_prediction"] == labels)
                thresholds = data["old_score"][max_old].astype(float)-data["current_score"][max_old].astype(float)
                needed = old * 20 * 23 // 100
                adjusted, bias = None, None
                if len(thresholds) >= needed:
                    bias = float(np.sort(thresholds)[-needed])
                    adjusted = analysis.measure(data, test, bias)
                margin, correct_old, correct_new = score.correct(data, labels, old)
                old_n = int((test & (labels < old)).sum())
                new_n = int(test.sum())-old_n
                old_ok = test & correct_old
                fixed_old = None
                if 5*int(old_ok.sum()) >= old_n:
                    n_old, n_new = score.at_old_target(margin, old_ok, test & correct_new, old_n, 20)
                    fixed_old = dict(old=100*n_old/old_n, new=100*n_new/new_n)
                row = {"task": task+1, "seed": seed, "method": method, "raw": raw, "offset": bias,
                       "group_auroc": score.group_auroc(data, test, old), "adjusted": adjusted,
                       "fixed_old": fixed_old}
                records.append(row)
                measured.append(raw["new_correct"])
            b, c, d = measured
            raw_differences.append(d-b)
            gains.append(d-c)
            required.append(b-c)
        surplus = [2*g-r for g,r in zip(gains,required)]
        diffs_pp = [d / 10 for d in raw_differences]
        summary[task+1] = {
            "raw_new_relocated_minus_replayed_pp": stats(diffs_pp),
            "raw_new_relocated_minus_neither_pp": stats([g/10 for g in gains]),
            "raw_new_replayed_minus_neither_pp": stats([r/10 for r in required]),
            "recovery_seeds": sum(d >= 0 for d in surplus),
            "half_recovery_supported": sum(surplus) >= 0 and sum(d >= 0 for d in surplus) >= 3,
            "within_five_seeds": sum(abs(d) <= 50 for d in raw_differences),
            "mean_absolute_difference": mean(map(abs, diffs_pp)),
            "closeness_supported": (sum(abs(d) for d in raw_differences) <= 250
                                    and sum(abs(d) <= 50 for d in raw_differences) >= 4),
        }
        b = [r["fixed_old"] for r in records if r["task"] == task+1 and r["method"] == METHODS[0]]
        d = [r["fixed_old"] for r in records if r["task"] == task+1 and r["method"] == METHODS[-1]]
        comparable = [(first,second) for first,second in zip(b,d) if first is not None and second is not None]
        summary[task+1]["matched_old20_relocated_minus_replayed"] = [
            dict(old_replayed=first["old"], old_relocated=second["old"], new_difference=second["new"]-first["new"])
            for first,second in comparable
        ]
        for method in METHODS:
            rows = [r for r in records if r["task"] == task+1 and r["method"] == method]
            eligible = [r["adjusted"] for r in rows if r["adjusted"] is not None]
            summary[task+1][method] = {
                "new": stats([r["raw"]["new_pct"] for r in rows]),
                "old": stats([r["raw"]["old_pct"] for r in rows]),
                "within_new": stats([r["raw"]["current10_pct"] for r in rows]),
                "group_auroc": stats([r["group_auroc"] for r in rows]),
                "adjusted": [{"new": r["new_pct"], "old": r["old_pct"]} for r in eligible],
            }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"tasks": summary, "records": records}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for task, item in summary.items():
        print(task, item["raw_new_relocated_minus_replayed_pp"],
              item["half_recovery_supported"], item["closeness_supported"])


if __name__ == "__main__":
    main()
