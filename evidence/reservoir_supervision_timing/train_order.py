"""Compare both orders of the same CIFAR-10 group-loss weights."""
import argparse
import copy
import json
from pathlib import Path
import time

from train_timing import ARTIFACT, config_for, engine, log, run_task, task_batches, torch

SEEDS = range(5201, 5211)


def coefficients(task, cfg, memory):
    seen = memory.seen
    arriving = 0
    profile = []
    for ids in task_batches(task, cfg):
        n = len(ids)
        replay = cfg["replay_batch"]
        profile.append(replay*arriving/((seen+arriving)*(n+replay)))
        arriving += n
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omission-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    base = json.loads((ARTIFACT/"configs/r9_followup/split_cifar10_classil_er_s201.json").read_text())
    args.output.mkdir(parents=True)
    records = []
    began = time.monotonic()
    for seed in SEEDS:
        cfg = config_for(base, seed)
        tasks, stream = engine.make_stream(cfg, str(args.data_root))
        reference = engine.local_model(cfg, cfg["init_seed"], device)
        memory = engine.Reservoir(cfg["memory_size"], seed+77001)
        run_task(reference, memory, tasks, 0, cfg, "retrieval", device, None)
        for task_id in range(1, len(tasks)):
            task = tasks[task_id]
            increasing = coefficients(task, cfg, memory)
            decreasing = increasing[:-1][::-1]+increasing[-1:]
            alternatives = [("omit", None)] if args.omission_only else [
                ("increasing", increasing), ("decreasing", decreasing)]
            for policy, schedule in alternatives+[("retrieval", None)]:
                model = reference
                branch = memory
                if policy != "retrieval":
                    model = copy.deepcopy(reference)
                    branch = copy.deepcopy(memory)
                output = args.output/f"s{seed}_t{task_id+1}_{policy}"
                output.mkdir()
                start = time.monotonic()
                measurements = run_task(model, branch, tasks, task_id, cfg, policy, device,
                                        output, schedule=schedule)
                if seed == SEEDS.start and task_id == 1:
                    seconds = time.monotonic()-start
                    log(policy=policy, task_seconds=seconds)
                records.append(dict(seed=seed, task=task_id+1, policy=policy,
                                    dose_by_update=measurements, coefficients=schedule,
                                    class_order=stream["class_order_original_ids"]))
        engine.atomic_json(args.output/"training.json", records)
        log(seed=seed, status="trained")
    log(status="complete", seconds=time.monotonic()-began)


if __name__ == "__main__":
    main()
