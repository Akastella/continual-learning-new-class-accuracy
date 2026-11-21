"""Measure the previous task after both weight schedules receive the same further training."""
import argparse
import copy
import json
from pathlib import Path
import time

from train_c100 import data, evaluate, log, model_for, np, plan, run_task, torch
from train_order_c100 import SEEDS, reordered_labels, schedules

TASKS = (5, 10, 15)
MEASUREMENTS = (25, 100, 250)


def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    source_train = data(args.data_root, device, True)
    source_test = data(args.data_root, device, False)
    args.output.mkdir(parents=True)
    records = []
    start = time.monotonic()
    for seed in args.seeds:
        order = np.random.default_rng(seed+40000).permutation(100)
        train = reordered_labels(source_train, order)
        test = reordered_labels(source_test, order)
        stream = plan(train[2], seed, 10000)
        reference = model_for(seed, device)
        run_task(reference, optimizer(reference), train, test, 0, stream[0], "retrieval", None)
        for task in range(1, max(TASKS)):
            exposure = stream[task]
            increasing = schedules(task, len(exposure[0]))["increasing"]
            if task+1 in TASKS:
                follow = stream[task+1]
                next_weights = schedules(task+1, len(follow[0]))["increasing"]
                descending = schedules(task, len(exposure[0]))["decreasing"]
                exchanged = schedules(task, len(exposure[0]), True)["exchanged"]
                for policy, weights in (("decreasing", descending), ("exchanged", exchanged)):
                    branch = copy.deepcopy(reference)
                    run_task(branch, optimizer(branch), train, test, task, exposure,
                             policy, None, weights)
                    folder = args.output/f"s{seed}_t{task+1}_{policy}"
                    folder.mkdir()
                    evaluate(branch, test, task, folder/"0.npz")
                    updater = optimizer(branch)
                    previous = 0
                    for update in MEASUREMENTS:
                        run_task(branch, updater, train, test, task+1, follow, "increasing",
                                 None, next_weights, start_update=previous, stop_update=update)
                        evaluate(branch, test, task, folder/f"{update}.npz")
                        previous = update
                    records.append(dict(seed=seed, task=task+1, policy=policy,
                                        class_order=order.tolist(),
                                        coefficients=weights, next_weights=next_weights,
                                        dose_by_update={update: sum(weights)+sum(next_weights[:update])
                                                        for update in (0, *MEASUREMENTS)}))
            if task+1 < max(TASKS):
                run_task(reference, optimizer(reference), train, test, task, exposure,
                         "increasing", None, increasing)
        (args.output/"training.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        log(seed=seed, seconds=time.monotonic()-start)
    log(status="complete", seconds=time.monotonic()-start)


if __name__ == "__main__":
    main()
