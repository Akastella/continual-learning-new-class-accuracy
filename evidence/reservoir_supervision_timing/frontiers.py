"""Compare new accuracy at the same achievable old accuracy early and late."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from recompute import load, stats


def frontier(data):
    labels = data["labels"]
    new = np.isin(labels, data["new_classes"])
    old_ok = ~new & (data["old_prediction"] == labels)
    new_ok = new & (data["new_prediction"] == labels)
    margin = data["old_score"].astype(float)-data["new_score"].astype(float)
    old_scores = np.sort(margin[old_ok])
    new_scores = np.sort(margin[new_ok])
    old_n = int((~new).sum())
    required = (np.arange(101)*old_n + 99)//100
    reachable = required <= len(old_scores)
    points = np.full((101,2), np.nan)
    points[0] = 0, 100*len(new_scores)/new.sum()
    q = np.flatnonzero(reachable & (required > 0))
    thresholds = old_scores[-required[q]]
    points[q,0] = 100*(len(old_scores)-np.searchsorted(old_scores,thresholds,side="left"))/old_n
    points[q,1] = 100*np.searchsorted(new_scores,thresholds,side="left")/new.sum()
    return points


def compare(root, early, policies=("retrieval", "first_presentation")):
    training = json.loads((root/"training.json").read_text())
    keys = sorted({(row["seed"],row["task"]) for row in training})
    endpoint = max(int(k) for k in training[0]["dose_by_update"])
    steps = (early, endpoint)
    scores = {}
    for seed,task in keys:
        for policy in policies:
            folder = root/f"s{seed}_t{task}_{policy}"
            for step in steps:
                scores[seed,task,policy,step] = frontier(load(folder/f"{step}.npz"))
    rows = []
    for q in range(101):
        by_seed = {}
        tasks = {}
        shared = []
        for seed,task in keys:
            retrieval, early = ([scores[seed,task,policy,step][q] for step in steps]
                                for policy in policies)
            values = np.asarray([retrieval,early])
            if np.isnan(values).any():
                continue
            differences = (values[1,:,1]-values[0,:,1]).tolist()
            by_seed.setdefault(seed,[]).append(differences)
            tasks[task] = 1+tasks.get(task,0)
            shared.append(dict(seed=seed,task=task,early=differences[0],late=differences[1],
                               achieved_old=values[:,:,0].tolist()))
        row = dict(target=q, pairs=len(shared), available_seeds=len(by_seed), tasks=tasks, shared=shared)
        if shared:
            for i,name in enumerate(("early","late")):
                means = [mean(z[i] for z in pairs) for pairs in by_seed.values()]
                row[name] = stats(means) if len(means)>1 else {"mean":means[0],"sample_sd":None,"seeds":means}
        rows.append(row)
    return dict(total_pairs=len(keys), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--early-update",type=int,default=16)
    parser.add_argument("--policies", nargs=2, default=("retrieval", "first_presentation"),
                        metavar=("REFERENCE", "COMPARISON"))
    args = parser.parse_args()
    result = compare(args.data, args.early_update, args.policies)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    for row in result["rows"]:
        if row["target"] in (20,30,70,80) and row["pairs"]:
            print(row["target"],row["pairs"],row["tasks"],
                  [(name,round(row[name]["mean"],3),row[name]["sample_sd"]) for name in ("early","late")])


if __name__ == "__main__":
    main()
