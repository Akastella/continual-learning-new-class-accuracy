"""Official CIFAR-100 train-only data and deterministic replay/augmentation.

This module deliberately provides no test loader. The archive is hashed in
full, but only the train and meta payloads are extracted and deserialized.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import tarfile
import time
import urllib.request

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz"
ARCHIVE_NAME = "cifar-100-python.tar.gz"
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
MEMBER_MD5 = {"train": "16019d7e3df5f24257cddd939b257f8d", "meta": "7973b15100ade9c7d40fb424638fde48"}
UPSTREAM_DATA_SHA256 = "087cba1fe9f2fc65aa3625588956d8d940592628c324d935b9e2dd2b8bbd5300"
MEAN = (.5071, .4867, .4408)
STD = (.2675, .2565, .2761)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=.5),
    transforms.ColorJitter(brightness=63/255), transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def event_seed(seed, *key):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Invalid seed")
    payload = json.dumps([seed, *key], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def verify_transform_reference():
    reference = ROOT / "third_party/pycil_reference/data.py"
    if file_digest(reference) != UPSTREAM_DATA_SHA256:
        raise ValueError("Pinned PyCIL data reference differs")


def prepare_train_files(directory, download=False, timeout_seconds=900):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE_NAME
    if not archive.exists():
        if not download:
            raise FileNotFoundError("Official CIFAR-100 archive missing; explicit download required")
        temporary = archive.with_suffix(archive.suffix + ".partial")
        started = time.monotonic()
        with temporary.open("xb") as destination:
            request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "R9-train-only-resource-witness"})
            with urllib.request.urlopen(request, timeout=60) as source:
                while block := source.read(4*1024*1024):
                    if time.monotonic() - started > timeout_seconds:
                        raise TimeoutError("Official download exceeded bounded duration; partial retained")
                    destination.write(block)
        if file_digest(temporary, "md5") != ARCHIVE_MD5:
            raise ValueError("Official archive MD5 differs; partial retained")
        temporary.rename(archive)
    if file_digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Official archive MD5 differs")
    extracted = directory / "cifar-100-python"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for name, expected in MEMBER_MD5.items():
            target = extracted / name
            if not target.exists():
                member = bundle.getmember("cifar-100-python/" + name)
                if not member.isfile() or member.size > 200_000_000:
                    raise ValueError("Unexpected official train/meta member")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("Missing selected train/meta payload")
                with source, target.open("xb") as destination:
                    while block := source.read(4*1024*1024):
                        destination.write(block)
            if file_digest(target, "md5") != expected:
                raise ValueError(f"Official {name} MD5 differs")
    return archive, extracted


@dataclass(frozen=True)
class TrainData:
    images: np.ndarray
    labels: np.ndarray
    fine_label_names: tuple
    provenance: dict


def load_train(directory, download=False):
    verify_transform_reference()
    archive, extracted = prepare_train_files(directory, download=download)
    with (extracted / "train").open("rb") as source:
        raw = pickle.load(source, encoding="bytes")
    with (extracted / "meta").open("rb") as source:
        metadata = pickle.load(source, encoding="bytes")
    image_rows = raw[b"data"]
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if image_rows.shape != (50000, 3072) or image_rows.dtype != np.uint8:
        raise ValueError("Official train image schema differs")
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 500)):
        raise ValueError("Official train fine labels are not 100 balanced classes")
    names = tuple(x.decode("utf-8") for x in metadata[b"fine_label_names"])
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError("Official fine label names differ")
    images = np.ascontiguousarray(image_rows.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    images.flags.writeable = False
    labels.flags.writeable = False
    provenance = {
        "dataset": "official_CIFAR100_train_only", "archive_url": ARCHIVE_URL,
        "archive_md5": ARCHIVE_MD5, "archive_sha256": file_digest(archive),
        "train_sha256": file_digest(extracted / "train"), "meta_sha256": file_digest(extracted / "meta"),
        "images_uint8_hwc_sha256": array_digest(images), "fine_labels_int64_sha256": array_digest(labels),
        "n": 50000, "shape": list(images.shape), "dtype": "uint8", "fine_labels_per_class": 500,
        "loaded_payloads": ["train", "meta"], "test_payload_loaded": False,
        "archive_scope": "Complete official archive was downloaded/hashed; test payload never extracted or deserialized.",
    }
    return TrainData(images, labels, names, provenance)


def class_mapping(seed):
    event_seed(seed, "validate")
    order = np.random.Generator(np.random.PCG64(seed + 40000)).permutation(100)
    inverse = np.empty(100, dtype=np.int64)
    inverse[order] = np.arange(100)
    return order, inverse


def task_indices(labels, seed, task):
    if not 0 <= task < 10:
        raise ValueError("task must be 0..9")
    order, _ = class_mapping(seed)
    return np.flatnonzero(np.isin(labels, order[10*task:10*task+10])).astype(np.int64)


def memory_indices(labels, seed, task, per_class):
    if not 0 <= task <= 10 or per_class not in (20, 200):
        raise ValueError("Only task 0..10 and memory20/200 are defined")
    order, _ = class_mapping(seed)
    parts = []
    for original_class in order[:10*task]:
        candidates = np.flatnonzero(labels == original_class).astype(np.int64)
        if len(candidates) != 500:
            raise ValueError("Each official class must have 500 training images")
        generator = np.random.Generator(np.random.PCG64(event_seed(seed + 50000, "memory", int(original_class))))
        parts.append(generator.permutation(candidates)[:per_class])
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def replay_count(consumed_current, current_n, memory_n, allocation, current_epoch_n=5000):
    if not 0 <= consumed_current < current_epoch_n or not 1 <= current_n <= current_epoch_n-consumed_current:
        raise ValueError("Current batch outside epoch")
    if memory_n < 0 or allocation not in ("current", "fixed", "proportional"):
        raise ValueError("Invalid replay allocation")
    if memory_n == 0 or allocation == "current":
        return 0
    if allocation == "fixed":
        return current_n
    return (consumed_current+current_n)*memory_n//current_epoch_n - consumed_current*memory_n//current_epoch_n


def replay_sequence(memory, count, seed, task, epoch):
    memory = np.asarray(memory, dtype=np.int64)
    if count < 0 or (count and not len(memory)) or len(np.unique(memory)) != len(memory):
        raise ValueError("Replay needs a unique nonempty memory")
    generator = np.random.Generator(np.random.PCG64(event_seed(seed + 60000, "replay-order", task, epoch)))
    parts = []
    remaining = count
    while remaining:
        permutation = generator.permutation(memory)
        take = min(remaining, len(memory))
        parts.append(permutation[:take])
        remaining -= take
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def epoch_plan(labels, seed, task, epoch, per_class, allocation):
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    current = task_indices(labels, seed, task)
    if len(current) != 5000:
        raise ValueError("Each official task must contain exactly 5000 training images")
    generator = np.random.Generator(np.random.PCG64(event_seed(seed + 30000, "current-order", task, epoch)))
    current = generator.permutation(current)
    memory = memory_indices(labels, seed, task, per_class)
    total_replay = 0 if not len(memory) or allocation == "current" else (5000 if allocation == "fixed" else len(memory))
    replay = replay_sequence(memory, total_replay, seed, task, epoch)
    rows, replay_offset = [], 0
    for step, offset in enumerate(range(0, 5000, 64)):
        current_ids = current[offset:offset+64]
        nr = replay_count(offset, len(current_ids), len(memory), allocation)
        rows.append({"step": step, "current_offset": offset, "replay_offset": replay_offset,
                     "current_indices": current_ids, "replay_indices": replay[replay_offset:replay_offset+nr],
                     "current_coefficient": len(current_ids)/(len(current_ids)+nr)})
        replay_offset += nr
    if replay_offset != total_replay:
        raise ValueError("Replay allocation does not close at epoch end")
    return rows, {"seed": seed, "task": task, "epoch": epoch, "per_class": per_class,
        "allocation": allocation, "steps": len(rows), "current_presentations": len(current),
        "replay_presentations": len(replay), "memory_n": len(memory),
        "class_order": class_mapping(seed)[0].tolist(), "current_order_sha256": array_digest(current),
        "memory_indices_sha256": array_digest(memory), "memory_indices": memory.tolist(),
        "replay_order_sha256": array_digest(replay), "current_indices": current.tolist(),
        "replay_indices": replay.tolist()}


def augmentation_seed(seed, task, epoch, source, position, sample_index):
    if source not in ("current", "replay", "extra_current"):
        raise ValueError("Unknown augmentation source")
    return event_seed(seed, "augmentation", task, epoch, source, position, sample_index)


def augment_image(image, random_seed):
    if image.dtype != np.uint8 or image.shape != (32, 32, 3):
        raise ValueError("Expected one uint8 CIFAR image")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(random_seed)
        return TRAIN_TRANSFORM(Image.fromarray(image))


def training_batch(dataset, row, seed, task, epoch):
    _, inverse = class_mapping(seed)
    tensors, targets, seeds = [], [], []
    for source in ("current", "replay"):
        for ordinal, index in enumerate(row[source+"_indices"]):
            index = int(index)
            if not 0 <= index < len(dataset.labels):
                raise ValueError("Training index is outside official train rows")
            target = int(inverse[dataset.labels[index]])
            if (source == "current" and not 10*task <= target < 10*task+10) or (source == "replay" and target >= 10*task):
                raise ValueError("Training batch source and mapped task label differ")
            key = augmentation_seed(seed, task, epoch, source, row[source+"_offset"]+ordinal, index)
            tensors.append(augment_image(dataset.images[index], key))
            targets.append(target)
            seeds.append(key)
    return torch.stack(tensors), torch.tensor(targets, dtype=torch.long), array_digest(np.asarray(seeds, dtype=np.int64))
