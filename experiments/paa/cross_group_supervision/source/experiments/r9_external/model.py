"""PyCIL ResNet-32 adapted to GN8 and one 100-class head.

The pinned upstream MIT-licensed source remains unmodified. This local model
is an explicit architecture adaptation, not a Cho et al. reproduction.
"""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "f3509b8ca3f20660ce4aa13f19d5283de81b4b35"
UPSTREAM_PATH = ROOT / "third_party/pycil_reference/cifar_resnet.py"
UPSTREAM_SHA256 = "cb24b7d3de97f28fc94205ac128a15465501b45e938449c6b60b329b8800ad91"
LICENSE_SHA256 = "0b349ecc68e75ab59493b80ff2308dc9fe144e6fcbb789ff62de427c4cea6920"
MODEL_NAME = "r9_cifar_resnet32_gn8"
INITIALIZATION = (
    "CPU default-generator seed inside fork_rng(devices=[]). Instantiate pinned "
    "upstream resnet32(), including its unused fc10 and upstream initialization. "
    "Conv weights: normal(0,sqrt(2/(kernel_height*kernel_width*out_channels))). "
    "Replace every BatchNorm2d with GroupNorm(8,C,eps=1e-5,affine=True), "
    "affine ones/zeros. Delete upstream unused fc10. Construct Linear(64,100), "
    "then kaiming_normal_(weight) with PyTorch defaults and zero bias. The "
    "discarded upstream fc10 and new Linear constructor consume CPU RNG before "
    "the final head initialization; preserve this order. Global CPU RNG restored."
)


def verify_reference():
    paths = {
        UPSTREAM_PATH: UPSTREAM_SHA256,
        UPSTREAM_PATH.with_name("LICENSE"): LICENSE_SHA256,
    }
    for path, expected in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned upstream reference differs: {path.name}")


@lru_cache(maxsize=1)
def _upstream_module():
    spec = importlib.util.spec_from_file_location("r9_pinned_pycil_cifar_resnet", UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned PyCIL reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(8, child.num_features, eps=1e-5, affine=True))
        else:
            _replace_batch_norm(child)


class CifarResNet32GN8(nn.Module):
    def __init__(self):
        super().__init__()
        verify_reference()
        self.backbone = _upstream_module().resnet32()
        _replace_batch_norm(self.backbone)
        del self.backbone.fc
        self.head = nn.Linear(64, 100)
        nn.init.kaiming_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        return self.head(self.backbone(images)["features"])


def make_model(seed, device="cpu"):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CifarResNet32GN8()
    return model.to(device)
