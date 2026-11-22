"""Compare window exchange at different old-accuracy targets on the same tasks."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def by_case(rows):
    return {(row["seed"], row["task"]): row for row in rows}


def summary(differences, cohort):
    grouped = {}
    for seed, task in sorted(cohort):
        grouped.setdefault(seed, []).append(differences[seed, task])
    values = [mean(group) for group in grouped.values()]
    return {"tasks": len(cohort), "per_seed": values, "mean": mean(values),
            "sd": stdev(values)}


def compare(folder):
    exchange = read(folder, "exchange_summary.json")
    reference = read(folder, "window_reference_retention.json")
    points = {}
    for target in (10, 20, 30):
        rows = exchange[f"measured_old{target}"]["steps"]["250"]["exchanged"]["pairs"]
        points[str(target)] = {key: row["comparison"]["new"] - row["reference"]["new"]
                               for key, row in by_case(rows).items()}
    reference_points = {}
    for name, result in reference.items():
        rows = by_case(result["test_curve"]["cases"])
        reference_points[name] = {
            key: row["scores"]["exchanged"]["new"] - row["scores"]["decreasing"]["new"]
            for key, row in rows.items()
        }
    highest = points["30"].keys()
    both = reference_points["decreasing"].keys() & reference_points["exchanged"].keys()
    return {
        "shared_30_percent_tasks": {target: summary(values, highest)
                                    for target, values in points.items()},
        "tasks_reaching_reference": {
            name: {"at_20": summary(points["20"], values.keys()),
                   "at_reference": summary(values, values.keys())}
            for name, values in reference_points.items()
        },
        "tasks_reaching_both_references": {
            "at_20": summary(points["20"], both),
            **{name: summary(values, both) for name, values in reference_points.items()}
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(args.data), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
