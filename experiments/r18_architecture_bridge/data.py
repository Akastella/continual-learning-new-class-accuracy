"""Direct loader for the official CIFAR-10 Python archive and R9 task split."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickle

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
TRAIN_FILES = tuple(f"data_batch_{i}" for i in range(1, 6))
TEST_FILE = "test_batch"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def tensor_sha256(value: torch.Tensor) -> str:
    return array_sha256(value.detach().cpu().contiguous().numpy())


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = path.read_bytes()
    raw = pickle.loads(payload, encoding="bytes")
    images = np.asarray(raw[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(raw[b"labels"], dtype=np.int64)
    if images.shape[1:] != (3, 32, 32) or len(images) != len(labels):
        raise ValueError(f"Invalid CIFAR-10 batch schema: {path}")
    return images, labels, {
        "path": path.as_posix(),
        "bytes": len(payload),
        "file_sha256": sha256_bytes(payload),
    }


def load_official(data_root: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, object]]:
    root = data_root / "r9_official" / "cifar-10-batches-py"
    train_parts = [_read_batch(root / name) for name in TRAIN_FILES]
    test_part = _read_batch(root / TEST_FILE)
    train_images = np.concatenate([part[0] for part in train_parts])
    train_labels = np.concatenate([part[1] for part in train_parts])
    test_images, test_labels = test_part[:2]
    if train_images.shape != (50000, 3, 32, 32) or test_images.shape != (10000, 3, 32, 32):
        raise ValueError("Official CIFAR-10 cardinality mismatch")
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split, images, labels in (
        ("train", train_images, train_labels),
        ("test", test_images, test_labels),
    ):
        x = (torch.from_numpy(images.copy()).float().div_(255.0) - mean) / std
        y = torch.from_numpy(labels.copy()).long()
        tensors[split] = x, y
    provenance = {
        "dataset": "CIFAR-10",
        "source": "official CIFAR-10 Python batches already present in the project data cache",
        "files": [part[2] for part in train_parts] + [test_part[2]],
        "splits": {
            "train": {
                "n": 50000,
                "raw_images_sha256": array_sha256(train_images),
                "raw_labels_sha256": array_sha256(train_labels),
                "normalized_images_sha256": tensor_sha256(tensors["train"][0]),
            },
            "test": {
                "n": 10000,
                "raw_images_sha256": array_sha256(test_images),
                "raw_labels_sha256": array_sha256(test_labels),
                "normalized_images_sha256": tensor_sha256(tensors["test"][0]),
            },
        },
    }
    return tensors, provenance


def make_tasks(data_root: Path, class_order_seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    tensors, provenance = load_official(data_root)
    class_order = np.random.default_rng(class_order_seed).permutation(10)
    inverse = np.argsort(class_order)
    tasks: list[dict[str, object]] = []
    for task_id in range(5):
        original_classes = class_order[2 * task_id : 2 * task_id + 2]
        task: dict[str, object] = {
            "id": task_id,
            "classes": [2 * task_id, 2 * task_id + 1],
            "original_classes": original_classes.tolist(),
            "splits": {},
        }
        for split in ("train", "test"):
            images, original_labels = tensors[split]
            ids = torch.where(
                (original_labels == int(original_classes[0]))
                | (original_labels == int(original_classes[1]))
            )[0]
            mapped_labels = torch.from_numpy(inverse).long()[original_labels[ids]]
            task[split] = (images[ids], mapped_labels, ids)
            task["splits"][split] = {
                "n": len(ids),
                "official_indices_sha256": tensor_sha256(ids),
                "images_sha256": tensor_sha256(images[ids]),
                "mapped_labels_sha256": tensor_sha256(mapped_labels),
            }
        tasks.append(task)
    provenance.update(
        class_order_seed=class_order_seed,
        class_order_original_ids=class_order.tolist(),
        original_class_to_global_output=inverse.tolist(),
    )
    return tasks, provenance

