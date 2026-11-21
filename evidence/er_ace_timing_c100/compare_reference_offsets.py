"""Compare loss schedules at the old accuracy of a reference schedule."""
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


def score_at_reference(rising, others, class_order):
    results = {}
    labels = rising["labels"]
    current = np.isin(labels, rising["new_classes"])
    val = validation_split(np.asarray(class_order)[labels])
    all_images = np.ones(len(labels), bool)
    for mode, fit, test in (("test_curve", all_images, all_images), ("validation_fit", val, ~val)):
        target = int(((rising["predicted"]==labels) & ~current & fit).sum())
        scores = {"reference": evaluate(rising["predicted"], labels, current, test)}
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
            results[mode] = dict(scores=scores, offsets=offsets)
    return results


def summarize(cases, total, policies, contrast, positive, negative, reference):
    result = {}
    for mode, rows in cases.items():
        seeds = sorted({row["seed"] for row in rows})
        values = {}
        for field in ("new","old"):
            by_seed = {
                label: [mean(r["scores"][label][field] for r in rows if r["seed"]==seed)
                        for seed in seeds] for label in policies
            }
            values[field] = {label: stats(series) for label, series in by_seed.items()}
            values[field][contrast] = stats([x-y for x,y in zip(by_seed[positive],by_seed[negative])])
        misses = {label: sum(r["scores"][label]["old"]+1e-10 < r["scores"][reference]["old"]
                             for r in rows) for label in policies if label!=reference}
        result[mode] = dict(common=len(rows), total=total, metrics=values, misses=misses, cases=rows)
    return result


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
        for mode, row in score_at_reference(rising, others, run["class_order"]).items():
            row["scores"] = {"increasing": row["scores"]["reference"], **{
                key: row["scores"][key] for key in others}}
            cases[mode].append(dict(seed=seed, task=task, **row))
    return summarize(cases,sum(r["policy"]=="omit" for r in runs),
                     ("increasing","decreasing","omit"),"omit_minus_reversal",
                     "omit","decreasing","increasing")


def compare_windows(reference, exchange):
    runs = json.loads((exchange/"training.json").read_text(encoding="utf-8"))
    windows = [row for row in runs if row["policy"]=="exchanged"]
    contrasts = {key: {"test_curve": [], "validation_fit": []}
                 for key in ("decreasing","exchanged")}
    for run in windows:
        seed, task = run["seed"], run["task"]
        path = f"s{seed}_t{task}"
        branches = {"decreasing":load(reference/f"{path}_decreasing"/"250.npz"),
                    "exchanged":load(exchange/f"{path}_exchanged"/"250.npz")}
        for base, other in (("decreasing","exchanged"),("exchanged","decreasing")):
            for mode, row in score_at_reference(branches[base], {other: branches[other]},
                                                run["class_order"]).items():
                row["scores"][base] = row["scores"].pop("reference")
                contrasts[base][mode].append(dict(seed=seed, task=task, **row))
    return {base:summarize(cases,len(windows),("decreasing","exchanged"),
                           "exchange_minus_reversal","exchanged","decreasing",base)
            for base,cases in contrasts.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--omission", type=Path)
    data.add_argument("--exchange", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.omission) if args.omission is not None else compare_windows(
        args.reference,args.exchange)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
