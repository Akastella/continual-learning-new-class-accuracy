"""Summarize paired single-pass continual-learning histories."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

POLICIES = ("replay", "incoming", "omit")
SEEDS = range(4101, 4111)


def stats(values):
    return {"mean": mean(values), "sample_sd": stdev(values), "values": values}


def compute(source):
    training = json.loads((source / "training.json").read_text(encoding="utf-8"))
    lookup = {(r["seed"], r["task"], r["policy"]): r for r in training}
    assert len(training) == len(lookup) == 150
    rows, results = [], {}
    for seed in SEEDS:
        for task in range(1, 6):
            reference = None
            for policy in POLICIES:
                with np.load(source / "reader_data" / f"s{seed}_{policy}_t{task}.npz", allow_pickle=False) as scores:
                    labels, current = scores["labels"], scores["new_classes"]
                    if reference is None:
                        reference = labels.copy(), current.copy()
                    else:
                        assert np.array_equal(labels, reference[0]) and np.array_equal(current, reference[1])
                    new = np.isin(labels, current)
                    correct = scores["predicted"] == labels
                    assert len(labels) == 2000*task and int(new.sum()) == 2000
                    exposure = lookup[seed, task, policy]
                    assert exposure["scheduled"] == 10000
                    assert exposure["replay"] == (4992 if task == 1 else 5024)
                    rows.append(dict(seed=seed, task=task, policy=policy,
                        new_correct=int(correct[new].sum()),
                        old_correct=int(correct[~new].sum()),
                        within_new_correct=int((scores["new_prediction"][new] == labels[new]).sum()),
                        replay=exposure["replay"], replay_current=exposure["replay_current"]))
    lookup = {(r["seed"], r["task"], r["policy"]): r for r in rows}
    for policy in POLICIES:
        summary = {}
        for name, value in (
            ("new", lambda r: r["new_correct"]/20),
            ("within_new", lambda r: r["within_new_correct"]/20),
            ("old", lambda r: r["old_correct"]/(20*(r["task"]-1))),
        ):
            summary[name] = stats([mean(value(lookup[s,t,policy]) for t in range(2,6)) for s in SEEDS])
        summary["new_per_task"] = {
            task: stats([lookup[s,task,policy]["new_correct"]/20 for s in SEEDS]) for task in range(1,6)}
        results[policy] = summary
    counts = {
        name: [sum(lookup[s,t,name]["new_correct"] for t in range(2,6)) for s in SEEDS]
        for name in POLICIES
    }
    replay_gain = [r-o for r,o in zip(counts["replay"], counts["omit"])]
    incoming_gain = [i-o for i,o in zip(counts["incoming"], counts["omit"])]
    placement = [i-r for i,r in zip(counts["incoming"], counts["replay"])]
    surplus = [2*i-r for i,r in zip(incoming_gain,replay_gain)]
    results.update(
        replay_gain_pp=stats([d/80 for d in replay_gain]),
        incoming_gain_pp=stats([d/80 for d in incoming_gain]),
        incoming_minus_replay_pp=stats([d/80 for d in placement]),
        replay_gain_pass=sum(replay_gain)>=8000 and sum(d>=800 for d in replay_gain)>=7,
        replay_pass_seeds=sum(d>=800 for d in replay_gain),
        half_recovery_pass=sum(surplus)>=0 and sum(d>=0 for d in surplus)>=7,
        half_recovery_seeds=sum(d>=0 for d in surplus),
        closeness_pass=sum(map(abs,placement))<=4000 and sum(abs(d)<=400 for d in placement)>=7,
        within_five_seeds=sum(abs(d)<=400 for d in placement),
        mean_absolute_difference_pp=mean(abs(d)/80 for d in placement))
    return {"results": results, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.data)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    r = result["results"]
    print({k:r[k] for k in ("replay_gain_pp","incoming_minus_replay_pp",
                            "replay_gain_pass","half_recovery_pass","closeness_pass")})


if __name__ == "__main__":
    main()
