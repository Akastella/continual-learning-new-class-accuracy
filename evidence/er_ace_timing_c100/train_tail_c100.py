"""Change the last 63 loss weights on a shared CIFAR-100 training path."""
import argparse
import copy
import json
from pathlib import Path
import time

from train_c100 import data, log, model_for, plan, run_task
from train_order_c100 import SEEDS, SUFFIX, reordered_labels, schedules
import numpy as np
import torch


def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    training_source = data(args.data_root, device, True)
    test_source = data(args.data_root, device, False)
    args.output.mkdir(parents=True)
    records = []
    started = time.monotonic()
    for seed in SEEDS:
        order = np.random.default_rng(seed+40000).permutation(100)
        training = reordered_labels(training_source, order)
        test = reordered_labels(test_source, order)
        stream = plan(training[2], seed, 10000)
        reference = model_for(seed, device)
        run_task(reference, make_optimizer(reference), training, test, 0, stream[0],
                 "retrieval", None)
        for task, exposure in enumerate(stream[1:], 1):
            start_task = time.monotonic()
            if start_task-started >= 3300:
                raise TimeoutError("The experiment exceeded 55 minutes.")
            count = len(exposure[0])
            shared_count = count-SUFFIX
            profiles = schedules(task, count)
            descending = profiles["decreasing"]
            increasing = profiles["increasing"]
            large = increasing[-SUFFIX:]
            shared = descending[:shared_count]
            branch_start = copy.deepcopy(reference)
            run_task(branch_start, make_optimizer(branch_start), training, test, task,
                     exposure, "decreasing", None, descending, stop_update=shared_count)
            tails = {"decreasing": descending, "large_increasing": shared+large,
                     "large_decreasing": shared+large[::-1]}
            for policy, profile in tails.items():
                model = copy.deepcopy(branch_start)
                output = args.output/f"s{seed}_t{task+1}_{policy}"
                output.mkdir()
                dose = run_task(model, make_optimizer(model), training, test, task,
                                exposure, policy, output, profile, start_update=shared_count)
                records.append(dict(seed=seed, task=task+1, policy=policy, coefficients=profile,
                                    class_order=order.tolist(), dose_by_update=dose))
            run_task(reference, make_optimizer(reference), training, test, task, exposure,
                     "increasing", None, increasing)
            if seed == SEEDS.start and task == 1:
                seconds = time.monotonic()-start_task
                log(trial_seconds=seconds, conservative_seconds=seconds*95*1.45+60)
                if seconds*95*1.45+60 >= 3300:
                    raise TimeoutError("The measured runtime exceeds the experiment budget.")
        record_file = args.output/"training.json"
        temporary = record_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
        temporary.replace(record_file)
        log(seed=seed, elapsed_seconds=time.monotonic()-started)
    log(status="complete", seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
