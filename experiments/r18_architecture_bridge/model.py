"""Pinned-source CIFAR ResNet-32 adapted to GN8 and one ten-class head."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = ROOT / "third_party" / "pycil_reference" / "cifar_resnet.py"
UPSTREAM_SHA256 = "cb24b7d3de97f28fc94205ac128a15465501b45e938449c6b60b329b8800ad91"
LICENSE_SHA256 = "0b349ecc68e75ab59493b80ff2308dc9fe144e6fcbb789ff62de427c4cea6920"
UPSTREAM_COMMIT = "f3509b8ca3f20660ce4aa13f19d5283de81b4b35"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference() -> None:
    expected = {UPSTREAM_PATH: UPSTREAM_SHA256, UPSTREAM_PATH.with_name("LICENSE"): LICENSE_SHA256}
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"Pinned upstream reference differs: {path}")


@lru_cache(maxsize=1)
def _upstream_module():
    verify_reference()
    spec = importlib.util.spec_from_file_location("r18_pinned_pycil_cifar_resnet", UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned PyCIL CIFAR ResNet")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(8, child.num_features, eps=1e-5, affine=True))
        else:
            _replace_batch_norm(child)


class CifarResNet32GN8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _upstream_module().resnet32()
        _replace_batch_norm(self.backbone)
        del self.backbone.fc
        self.head = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)["features"])


def make_model(seed: int, device: torch.device) -> CifarResNet32GN8:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CifarResNet32GN8()
    return model.to(device).to(memory_format=torch.channels_last)

