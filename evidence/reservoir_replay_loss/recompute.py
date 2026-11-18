"""Calculate paired results for the CIFAR-10 reservoir replay losses."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ("all_class_replay", "current_two_replay")


def stats(values):
    return {"mean": mean(values), "sample_sd": stdev(values), "values": values}


def calculate(source):
    records = []
    training = json.loads((source / "training.json").read_text(encoding="utf-8"))
    for seed in range(201, 211):
        earlier_labels = None
        for condition in CONDITIONS:
            with np.load(source / "reader_data" / f"s{seed}_{condition}.npz", allow_pickle=False) as batch:
                assert int(batch["seed"]) == seed and str(batch["condition"]) == condition
                labels = batch["labels"]
                new = np.isin(labels, batch["new_classes"])
                assert len(labels) == 10000 and int(new.sum()) == 2000
                if earlier_labels is None:
                    earlier_labels = labels.copy()
                else:
                    assert np.array_equal(earlier_labels, labels)
                old_score, new_score = batch["old_score"], batch["new_score"]
                old_wins = ((old_score > new_score)
                            | ((old_score == new_score) & (batch["old_prediction"] < batch["new_prediction"])))
                reconstructed = np.where(old_wins, batch["old_prediction"], batch["new_prediction"])
                assert np.array_equal(reconstructed, batch["predicted"])
                correct = reconstructed == labels
                gap = old_score.astype(float) - new_score.astype(float)
                old_gaps = np.sort(gap[~new])
                left = np.searchsorted(old_gaps, gap[new], side="left")
                right = np.searchsorted(old_gaps, gap[new], side="right")
                auroc = float(100 * (1 - ((left + right) / (2 * len(old_gaps))).mean()))
                record = next(r for r in training if (r["seed"], r["condition"]) == (seed, condition))
                records.append(dict(seed=seed, condition=condition,
                    new_correct=int(correct[new].sum()), old_correct=int(correct[~new].sum()),
                    within_new_correct=int((batch["new_prediction"][new] == labels[new]).sum()),
                    auroc_pct=auroc, current_replay_presentations=record["replayed_current"],
                    total_replay_presentations=record["counts"]["replay_examples"]))
    result = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        result[condition] = {
            "new_accuracy": stats([r["new_correct"]/20 for r in rows]),
            "old_accuracy": stats([r["old_correct"]/80 for r in rows]),
            "within_new_accuracy": stats([r["within_new_correct"]/20 for r in rows]),
            "group_auroc": stats([r["auroc_pct"] for r in rows]),
        }
    left, right = ([r for r in records if r["condition"] == c] for c in CONDITIONS)
    differences = [a["new_correct"]-b["new_correct"] for a,b in zip(left,right)]
    result["new_advantage"] = stats([d / 20 for d in differences])
    result["old_difference"] = stats([(a["old_correct"]-b["old_correct"])/80 for a,b in zip(left,right)])
    result["within_new_difference"] = stats([(a["within_new_correct"]-b["within_new_correct"])/20 for a,b in zip(left,right)])
    result["auroc_difference"] = stats([a["auroc_pct"]-b["auroc_pct"] for a,b in zip(left,right)])
    result["at_least_ten_pp_seeds"] = sum(d >= 200 for d in differences)
    result["prediction_supported"] = sum(differences) >= 2000 and result["at_least_ten_pp_seeds"] >= 7
    return {"summary": result, "counts": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=HERE)
    args = parser.parse_args()
    result = calculate(args.data)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["summary"]["new_advantage"], result["summary"]["prediction_supported"])


if __name__ == "__main__":
    main()
