"""Create fixed configs and execute the two official CBP parameter profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(201, 206))
PROFILES = {
    "nature_cifar": {"replacement_rate": 1e-5, "maturity_threshold": 1000, "decay_rate": 0.99},
    "official_conv_default": {"replacement_rate": 1e-4, "maturity_threshold": 100, "decay_rate": 0.9},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(seed: int, profile: str) -> dict:
    source = ROOT / "configs/r9_followup" / f"split_cifar10_classil_sgd_s{seed}.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    cfg = {
        **original,
        "run_type": "posthoc_official_cbp_baseline",
        "method": "cbp",
        "profile": profile,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "cbp_rng_seed": 180000 + seed,
        "primary_source_config": source.relative_to(ROOT).as_posix(),
        "primary_source_config_sha256": sha(source),
        "official_upstream_commit": "a6b79580d85f3025bdb601566d3627c5f489f13b",
        **PROFILES[profile],
    }
    return cfg


def write_configs() -> list[tuple[Path, Path]]:
    config_root = ROOT / "configs/r18_cbp_official"
    config_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for profile in PROFILES:
        for seed in SEEDS:
            key = f"{profile}_s{seed}"
            path = config_root / f"{key}.json"
            payload = json.dumps(config(seed, profile), indent=2, allow_nan=False) + "\n"
            if path.exists() and path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"Existing config drift: {path}")
            if not path.exists():
                path.write_text(payload, encoding="utf-8")
            runs.append((path, ROOT / "results/r18_cbp_official" / key))
    manifest = {
        "schema": "r18_official_cbp_manifest_v1",
        "seeds": list(SEEDS),
        "profiles": PROFILES,
        "configs": {p.relative_to(ROOT).as_posix(): sha(p) for p, _ in runs},
        "upstream_commit": "a6b79580d85f3025bdb601566d3627c5f489f13b",
    }
    manifest_path = config_root / "manifest.json"
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != payload:
        raise ValueError("Existing manifest drift")
    if not manifest_path.exists():
        manifest_path.write_text(payload, encoding="utf-8")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    args = parser.parse_args()
    runs = write_configs()
    for config_path, output in runs:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if args.profile != "all" and cfg["profile"] != args.profile:
            continue
        result = output / "result.json"
        if result.exists() and json.loads(result.read_text(encoding="utf-8"))["status"] == "complete":
            print(json.dumps({"skip_complete": output.relative_to(ROOT).as_posix()}), flush=True)
            continue
        command = [sys.executable, "-u", "-m", "experiments.r18_cbp_official.engine",
                   "--config", str(config_path), "--data-root", "data",
                   "--output", output.relative_to(ROOT).as_posix()]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
