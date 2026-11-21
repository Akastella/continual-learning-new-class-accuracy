"""Compare the order of equal group-loss weights on random CIFAR-100 streams."""
import argparse
import copy
import json
from pathlib import Path
import time

from train_c100 import data, log, model_for, np, plan, run_task, torch

SEEDS = range(7201, 7206)
SUFFIX = 63


def reordered_labels(inputs, order):
    images, _, labels = inputs
    mapped = np.argsort(order)[labels]
    return images, torch.as_tensor(mapped, device=images.device), mapped


def schedules(task, updates, exchange_windows=False, omit_group_loss=False):
    increasing = [j/(task*updates+j) for j in range(updates)]
    decreasing = increasing[::-1]
    if omit_group_loss:
        return {"omit": [0.]*updates, "increasing": increasing}
    if exchange_windows:
        exchanged = decreasing[-SUFFIX:]+decreasing[SUFFIX:-SUFFIX]+decreasing[:SUFFIX]
        return {"exchanged": exchanged, "increasing": increasing}
    return {
        "decreasing": decreasing,
        "reverse_prefix": increasing[:-SUFFIX][::-1]+increasing[-SUFFIX:],
        "increasing": increasing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--exchange-windows", action="store_true")
    comparison.add_argument("--omit-group-loss", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    train = data(args.data_root, device, True)
    test = data(args.data_root, device, False)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    record_file = args.output/"training.json"
    records = json.loads(record_file.read_text(encoding="utf-8")) if args.resume else []
    completed = {row["seed"] for row in records}
    expected = [(seed, task, policy) for seed in args.seeds if seed in completed
                for task in range(2, 21) for policy in schedules(task-1, 250, args.exchange_windows, args.omit_group_loss)]
    actual = [(row["seed"], row["task"], row["policy"]) for row in records]
    if actual != expected:
        raise ValueError("The existing training record does not contain complete seed runs.")
    for seed, task, policy in expected:
        folder = args.output/f"s{seed}_t{task}_{policy}"
        if not all((folder/f"{step}.npz").is_file() for step in (25,250)):
            raise FileNotFoundError(folder)
    started = time.monotonic()
    trial = []
    for seed in args.seeds:
        if seed in completed:
            continue
        order = np.random.default_rng(seed+40000).permutation(100)
        training = reordered_labels(train, order)
        evaluation = reordered_labels(test, order)
        stream = plan(training[2], seed, 10000)
        reference = model_for(seed, device)
        for task, exposure in enumerate(stream):
            if not task:
                optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                run_task(reference, optimizer, training, evaluation, task, exposure, "retrieval", args.output)
                continue
            profiles = schedules(task, len(exposure[0]), args.exchange_windows, args.omit_group_loss)
            for policy, weights in profiles.items():
                model = reference if policy == "increasing" else copy.deepcopy(reference)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                folder = args.output/f"s{seed}_t{task+1}_{policy}"
                folder.mkdir()
                began = time.monotonic()
                doses = run_task(model, optimizer, training, evaluation, task, exposure,
                                 policy, folder, time_weights=weights)
                if seed == args.seeds[0] and task == 1:
                    seconds = time.monotonic()-began
                    trial.append(seconds)
                    log(policy=policy, seconds=seconds)
                records.append(dict(seed=seed, task=task+1, policy=policy,
                                    dose_by_update=doses, coefficients=weights,
                                    class_order=order.tolist()))
            if seed == args.seeds[0] and task == 1:
                log(estimated_seconds=1.5*sum(trial)*19*len(args.seeds)+60)
        pending_record = record_file.with_suffix(".tmp")
        with pending_record.open("w", encoding="utf-8") as output:
            json.dump(records, output, indent=2)
        pending_record.replace(record_file)
        log(seed=seed, elapsed_seconds=time.monotonic()-started)
    log(status="complete", seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
