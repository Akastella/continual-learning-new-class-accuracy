"""Summarize the twenty-task timing experiment and original old-task forgetting."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"reservoir_supervision_timing"))
from recompute import compare, load, stats
from frontiers import frontier

TARGETS = (10,20,30)


def accuracy_by_task(data):
    labels = data["labels"]
    correct = data["predicted"] == labels
    task = labels//5
    n = task.max()+1
    return 100*np.bincount(task,weights=correct,minlength=n)/np.bincount(task,minlength=n)


def summary(data, first_task):
    result = compare(data,20)
    records = json.loads((data/"training.json").read_text())
    seeds = result["seeds"]
    tasks = result["tasks"]
    policies = sorted({r["policy"] for r in records})
    current = {}
    matrix = {}
    curves = {}
    for record in records:
        s,t,m = (record[k] for k in ("seed","task","policy"))
        folder = data/f"s{s}_t{t}_{m}"
        for step in (25,250):
            values = load(folder/f"{step}.npz")
            curves[s,t,m,step] = frontier(values)
            if step == 250:
                current[s,t,m] = accuracy_by_task(values).tolist()
    for m in policies:
        complete = []
        for s in seeds:
            first = accuracy_by_task(load(first_task/f"s{s}_t1.npz")).tolist()
            by_task = [first]+[current[s,t,m] for t in tasks]
            final = by_task[-1]
            forgotten = [max(values[t] for values in by_task[t:])-final[t]
                         for t in range(19)]
            complete.append(dict(seed=s,task_accuracy=by_task,old_forgetting=float(mean(forgotten)),
                                 mean_final_old=float(mean(final[:19])),
                                 worst_final_old=float(min(final[:19]))))
        matrix[m] = dict(per_seed=complete,
                         **{stat:stats([x[stat] for x in complete])
                            for stat in ("old_forgetting","mean_final_old","worst_final_old")})
    result["task_results"] = matrix
    last = {
        m: {
            "new": [current[s, tasks[-1], m][-1] for s in seeds],
            "old": [mean(current[s, tasks[-1], m][:-1]) for s in seeds],
            "overall": [mean(current[s, tasks[-1], m]) for s in seeds],
        }
        for m in policies
    }
    result["last_task"] = {
        m: {
            "means": {field: stats(values) for field, values in scores.items()},
            "paired": {
                field: stats([a-b for a, b in zip(values, last["retrieval"][field])])
                for field, values in scores.items()
            },
        }
        for m, scores in last.items()
    }
    matched = {}
    for m in policies:
        by_q = {}
        for q in TARGETS:
            available = []
            for s in seeds:
                for t in tasks:
                    a = np.array([curves[s,t,"retrieval",step][q] for step in (25,250)])
                    b = np.array([curves[s,t,m,step][q] for step in (25,250)])
                    if np.isnan(a).any() or np.isnan(b).any():
                        continue
                    d = b[:,1]-a[:,1]
                    available.append(dict(seed=s,task=t,differences=d.tolist(),
                                          old_reference=a[:,0].tolist(),old_placement=b[:,0].tolist()))
            row = dict(common=len(available),per_pair=available,
                       coverage_by_task={t:sum(x["task"]==t for x in available) for t in tasks})
            included = [s for s in seeds if any(x["seed"]==s for x in available)]
            if len(included)>1:
                row["drops"] = {
                    step:stats([mean(x["differences"][i] for x in available if x["seed"]==s) for s in included])
                    for i,step in enumerate((25,250))
                }
            by_q[q] = row
        matched[m] = by_q
    result["matched_early_late"] = matched
    result["per_task"] = {}
    for m in policies:
        result["per_task"][m] = {}
        for t in tasks:
            available = [x["differences"][1] for x in matched[m][20]["per_pair"] if x["task"] == t]
            matched_new = {"mean": mean(available), "sample_sd": None} if available else None
            if len(available) > 1:
                matched_new = stats(available)
            result["per_task"][m][t] = {
                "raw_new": stats([current[s,t,m][-1] for s in seeds]),
                "paired_raw_new": result["summaries"][250][m]["by_task"][t]["new"],
                "paired_at_old20": matched_new,
                "matched_old20_n": len(available),
            }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--first-task",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=summary(args.data,args.first_task)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,allow_nan=False,indent=2)
        stream.write("\n")
    for mode,record in result["summaries"][250].items():
        print(mode, "new", record["paired"]["new"]["mean"], record["paired"]["new"]["sample_sd"],
              "old", record["paired"]["old"]["mean"], "forgetting", result["task_results"][mode]["old_forgetting"]["mean"])


if __name__=="__main__":
    main()
