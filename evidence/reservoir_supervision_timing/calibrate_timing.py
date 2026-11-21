"""Fit new-score offsets on separate validation identities in the timing runs."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from recompute import load, stats


def apply(path,target):
    data = load(path)
    labels = data["labels"]
    new = np.isin(labels,data["new_classes"])
    split = np.zeros(len(labels),dtype=bool)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        generator = np.random.default_rng(2026111807 + int(label))
        split[generator.permutation(rows)[:len(rows)//5]] = True
    old_ok = ~new & (data["old_prediction"] == labels)
    new_ok = new & (data["new_prediction"] == labels)
    gap = data["old_score"].astype(float)-data["new_score"].astype(float)
    required = ((target+3)*int((split & ~new).sum())+99)//100
    eligible = np.sort(gap[split & old_ok])
    if len(eligible) < required:
        return None
    offset = eligible[-required]
    validation_old = (split & old_ok & (gap>=offset)).sum()/(split & ~new).sum()*100
    test_old = (~split & old_ok & (gap>=offset)).sum()/(~split & ~new).sum()*100
    test_new = (~split & new_ok & (gap<offset)).sum()/(~split & new).sum()*100
    return dict(offset=float(offset), validation_old=float(validation_old),
                old=float(test_old), new=float(test_new), missed=bool(test_old<target))


def compare(root,target,reference="retrieval"):
    records = json.loads((root/"training.json").read_text())
    results = {}
    for row in records:
        seed,task,policy = row["seed"],row["task"],row["policy"]
        directory = root/f"s{seed}_t{task}_{policy}"
        for step in row["dose_by_update"]:
            results[seed,task,policy,step] = apply(directory/f"{step}.npz",target)
    comparisons = {}
    steps = sorted({key[3] for key in results},key=int)
    policies = sorted({key[2] for key in results})
    keys = sorted({key[:2] for key in results})
    for step in steps:
        options = {}
        for policy in policies:
            baseline = {pair: results[(*pair,reference,step)] for pair in keys}
            alternative = {pair: results[(*pair,policy,step)] for pair in keys}
            common = []
            by_seed = {}
            for pair in keys:
                first,second = baseline[pair],alternative[pair]
                if first is None or second is None:
                    continue
                row = dict(seed=pair[0],task=pair[1],retrieval=first,alternative=second)
                common.append(row)
                by_seed.setdefault(pair[0],[]).append(row)
            paired = {}
            if len(by_seed)>1:
                for field in ("old","new"):
                    paired[field] = stats([mean(x["alternative"][field]-x["retrieval"][field] for x in values)
                                           for values in by_seed.values()])
            options[policy] = dict(
                total=len(keys),
                available_retrieval=sum(x is not None for x in baseline.values()),
                available_alternative=sum(x is not None for x in alternative.values()),
                common=len(common),
                common_by_task={task:sum(x["task"]==task for x in common) for task in sorted({t for _,t in keys})},
                misses=[sum(x[k]["missed"] for x in common) for k in ("retrieval","alternative")],
                paired=paired,paired_rows=common)
        comparisons[step] = options
    return dict(old_target=target, validation_margin=3, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--old-target",type=int,required=True)
    parser.add_argument("--reference", default="retrieval")
    args = parser.parse_args()
    report = compare(args.data,args.old_target,args.reference)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    for step,options in report["comparisons"].items():
        print(step,{policy:(row["common"],row["misses"],{k:round(v["mean"],3) for k,v in row["paired"].items()})
                    for policy,row in options.items()})


if __name__ == "__main__":
    main()
