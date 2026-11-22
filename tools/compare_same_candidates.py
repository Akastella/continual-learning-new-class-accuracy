"""Compare raw and adjusted validation selections over two candidate sets."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from recompute_score_frontier import all_names, calc, loadval, optimal


def select(root, seed, names, adjusted, q):
    feasible = []
    for name in names:
        scores, labels = loadval(root, seed, name)
        offset = optimal(scores, labels, q) if adjusted else 0.
        if offset is not None:
            validation = calc(scores, labels, offset)
            if validation["old"] >= q:
                feasible.append((validation["current"], name, offset))
    if not feasible:
        return None
    return min(feasible, key=lambda point: (-point[0], point[1]))[1:]


def evaluate(root, seed, dataset, choice):
    name, offset = choice
    folder = root / "evidence/score_frontier"
    file = folder / "diagnostics/test_logits" / f"{dataset}_s{seed}_{name}.npz"
    if not file.exists():
        file = folder / "test_logits" / file.name
    with np.load(file, allow_pickle=False) as data:
        metrics = calc(data["logits"].astype(float), data["labels"], offset)
        return dict(metrics, name=name, offset=offset)


def paired_change(before, after, seeds):
    result = {}
    for metric in ("current", "old"):
        differences = [after[seed][metric] - before[seed][metric] for seed in seeds]
        result[metric] = {"mean": mean(differences), "sd": stdev(differences)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    seeds = range(201, 211)
    for dataset in ("cifar101", "cifar102"):
        rows = {}
        for menu, names in (("two", ("er_standard", "sgd_standard")), ("thirty", all_names())):
            options = {}
            for adjusted in (False, True):
                choices = {seed: select(args.artifact, seed, names, adjusted, 30) for seed in seeds}
                options["adjusted" if adjusted else "raw"] = {
                    seed: evaluate(args.artifact, seed, dataset, choice)
                    for seed, choice in choices.items() if choice is not None
                }
            a, b = options["raw"], options["adjusted"]
            common = a.keys() & b.keys()
            minimum_old = {policy: sum(x["old"] < 30 for x in values.values())
                           for policy, values in options.items()}
            rows[menu] = dict(choices=options, common=sorted(common), misses=minimum_old,
                              new_change_common=mean(b[s]["current"]-a[s]["current"] for s in common),
                              old_change_common=mean(b[s]["old"]-a[s]["old"] for s in common),
                              misses_common=[sum(values[s]["old"] < 30 for s in common)
                                             for values in (a,b)])
            if menu == "thirty":
                fixed = {}
                for seed in sorted(common):
                    name = a[seed]["name"]
                    scores, labels = loadval(args.artifact, seed, name)
                    offset = optimal(scores, labels, 30)
                    assert offset is not None and calc(scores, labels, offset)["old"] >= 30
                    fixed[seed] = evaluate(args.artifact, seed, dataset, (name, offset))
                rows[menu]["fixed_choice"] = fixed
                rows[menu]["offset_and_reselection"] = {
                    "adjust_fixed_choice": paired_change(a, fixed, sorted(common)),
                    "reselect_after_adjustment": paired_change(fixed, b, sorted(common)),
                }
                rows[menu]["fixed_choice_misses"] = sum(values["old"] < 30
                                                        for values in fixed.values())
        result[dataset] = rows
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for dataset, rows in result.items():
        print(dataset, {k: {a:b for a,b in v.items() if a not in {"choices", "fixed_choice"}} for k,v in rows.items()})


if __name__ == "__main__":
    main()
