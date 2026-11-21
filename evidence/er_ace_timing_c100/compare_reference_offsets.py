"""Compare loss schedules at the old accuracy of the increasing schedule."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"reservoir_supervision_timing"))
from calibrate_timing import validation_split
from compare_tail_offsets import evaluate, predict
from recompute import load, stats


def compare(reference, omission):
    runs = json.loads((omission/"training.json").read_text(encoding="utf-8"))
    cases = {"test_curve": [], "validation_fit": []}
    for run in runs:
        if run["policy"] != "omit":
            continue
        seed, task = run["seed"], run["task"]
        step = str(max(map(int, run["dose_by_update"])))
        path = f"s{seed}_t{task}"
        rising = load(reference/f"{path}_increasing"/f"{step}.npz")
        others = {key: load(folder/f"{path}_{key}"/f"{step}.npz")
                  for key, folder in (("decreasing", reference), ("omit", omission))}
        labels = rising["labels"]
        current = np.isin(labels, rising["new_classes"])
        val = validation_split(np.asarray(run["class_order"])[labels])
        for mode, fit, test in (("test_curve", np.ones(len(labels), bool), np.ones(len(labels), bool)),
                                ("validation_fit", val, ~val)):
            target = int(((rising["predicted"]==labels) & ~current & fit).sum())
            scores = {"increasing": evaluate(rising["predicted"], labels, current, test)}
            offsets = {}
            for policy, data in others.items():
                assert np.array_equal(data["labels"], labels)
                assert np.array_equal(data["new_classes"], rising["new_classes"])
                margin = data["old_score"].astype(float)-data["new_score"].astype(float)
                eligible = np.sort(margin[fit & ~current & (data["old_prediction"]==labels)])
                if len(eligible) < target:
                    break
                offset = eligible[-target] if target else np.nextafter(margin[fit].max(), np.inf)
                offsets[policy] = float(offset)
                scores[policy] = evaluate(predict(data, offset), labels, current, test)
            else:
                cases[mode].append(dict(seed=seed, task=task, scores=scores, offsets=offsets))
    result = {}
    for mode, rows in cases.items():
        seeds = sorted({row["seed"] for row in rows})
        values = {}
        for field in ("new","old"):
            by_seed = {
                label: [mean(r["scores"][label][field] for r in rows if r["seed"]==seed)
                        for seed in seeds]
                for label in ("increasing", "decreasing", "omit")
            }
            values[field] = {label: stats(series) for label, series in by_seed.items()}
            values[field]["omit_minus_reversal"] = stats([x-y for x,y in zip(by_seed["omit"],by_seed["decreasing"])])
        misses = {label: sum(r["scores"][label]["old"]+1e-10 < r["scores"]["increasing"]["old"]
                             for r in rows) for label in ("decreasing","omit")}
        result[mode] = dict(common=len(rows), total=sum(r["policy"]=="omit" for r in runs),
                            metrics=values, misses=misses, cases=rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--omission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.omission)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for mode, values in result.items():
        print(mode, values["common"], values["total"],
              values["metrics"]["new"]["omit_minus_reversal"]["mean"])
