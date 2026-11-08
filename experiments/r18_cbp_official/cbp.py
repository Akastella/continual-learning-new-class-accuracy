"""Device-safe port of the official ConvGnT generate-and-test semantics.

The equations and update order follow ``lop/algos/convGnT.py`` at commit
``a6b79580d85f3025bdb601566d3627c5f489f13b``.  The upstream implementation
allocates its bookkeeping tensors on CPU even when ``device='cuda'``.  This
port places bookkeeping on the model device and uses a dedicated CPU random
generator for traceable replacement draws; neither change alters the CBP
selection, reset, or utility equations.
"""
from __future__ import annotations

from math import sqrt

import torch
from torch import nn


def layer_bound(layer: nn.Module, init: str, gain: float) -> float:
    """Exact ``lop.utils.miscellaneous.get_layer_bound`` formula."""
    if isinstance(layer, nn.Conv2d):
        return sqrt(1.0 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    if not isinstance(layer, nn.Linear):
        raise TypeError(type(layer).__name__)
    if init == "default":
        return sqrt(1.0 / layer.in_features)
    if init == "xavier":
        return gain * sqrt(6.0 / (layer.in_features + layer.out_features))
    if init == "lecun":
        return sqrt(3.0 / layer.in_features)
    return gain * sqrt(3.0 / layer.in_features)


class OfficialConvGnT:
    """Official ConvGnT logic specialized to the primary sequential CNN."""

    def __init__(
        self,
        weight_layers: list[nn.Module],
        *,
        replacement_rate: float,
        decay_rate: float,
        maturity_threshold: int,
        num_last_filter_outputs: int,
        rng_seed: int,
        device: torch.device,
        init: str = "kaiming",
        util_type: str = "contribution",
    ) -> None:
        if len(weight_layers) != 5:
            raise ValueError("Primary CNN must expose four hidden weight layers and one head")
        if util_type != "contribution":
            raise ValueError("This audit only ports the official contribution utility")
        if not 0.0 <= replacement_rate <= 1.0 or not 0.0 <= decay_rate < 1.0:
            raise ValueError("Invalid CBP rate")
        if maturity_threshold < 0:
            raise ValueError("Invalid maturity threshold")
        self.layers = weight_layers
        self.num_hidden_layers = len(weight_layers) - 1
        self.replacement_rate = float(replacement_rate)
        self.decay_rate = float(decay_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.num_last_filter_outputs = int(num_last_filter_outputs)
        self.device = device
        self.util_type = util_type
        widths = [layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features
                  for layer in weight_layers[:-1]]
        self.util = [torch.zeros(n, device=device) for n in widths]
        self.bias_corrected_util = [torch.zeros(n, device=device) for n in widths]
        self.ages = [torch.zeros(n, device=device) for n in widths]
        self.mean_feature_act = [torch.zeros(n, device=device) for n in widths]
        self.mean_abs_feature_act = [torch.zeros(n, device=device) for n in widths]
        self.accumulated = [0.0] * self.num_hidden_layers
        self.per_step_replacement = [self.replacement_rate * n for n in widths]
        gain = nn.init.calculate_gain("relu")
        self.bounds = [layer_bound(layer, init, gain) for layer in weight_layers[:-1]]
        self.rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        self.replacements = [0] * self.num_hidden_layers
        self.replacement_events: list[dict] = []
        self.steps = 0

    @torch.no_grad()
    def _update_utility(self, i: int, features: torch.Tensor) -> None:
        self.util[i].mul_(self.decay_rate)
        correction = 1.0 - self.decay_rate ** self.ages[i]
        current, following = self.layers[i], self.layers[i + 1]
        if isinstance(following, nn.Linear):
            output_weight_mag = following.weight.abs().mean(dim=0)
        else:
            output_weight_mag = following.weight.abs().mean(dim=(0, 2, 3))

        self.mean_feature_act[i].mul_(self.decay_rate)
        self.mean_abs_feature_act[i].mul_(self.decay_rate)
        if isinstance(current, nn.Linear):
            feature_mean = features.mean(dim=0)
            feature_abs_mean = features.abs().mean(dim=0)
        elif isinstance(following, nn.Conv2d):
            feature_mean = features.mean(dim=(0, 2, 3))
            feature_abs_mean = features.abs().mean(dim=(0, 2, 3))
        else:
            feature_mean = features.mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)
            feature_abs_mean = features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)
        self.mean_feature_act[i].add_((1.0 - self.decay_rate) * feature_mean)
        self.mean_abs_feature_act[i].add_((1.0 - self.decay_rate) * feature_abs_mean)

        # contribution utility in the official code does not center activation.
        if isinstance(following, nn.Linear) and isinstance(current, nn.Conv2d):
            new_utility = (output_weight_mag * features.abs().mean(dim=0)).view(
                -1, self.num_last_filter_outputs).mean(dim=1)
        elif isinstance(following, nn.Linear):
            new_utility = output_weight_mag * features.abs().mean(dim=0)
        else:
            new_utility = output_weight_mag * features.abs().mean(dim=(0, 2, 3))
        self.util[i].add_((1.0 - self.decay_rate) * new_utility)
        self.bias_corrected_util[i].copy_(self.util[i] / correction)

    def _outgoing_indices(self, i: int, selected: torch.Tensor) -> torch.Tensor:
        if isinstance(self.layers[i], nn.Conv2d) and isinstance(self.layers[i + 1], nn.Linear):
            offsets = torch.arange(self.num_last_filter_outputs, device=self.device)
            return (selected * self.num_last_filter_outputs).repeat_interleave(
                self.num_last_filter_outputs) + offsets.repeat(selected.numel())
        return selected

    @torch.no_grad()
    def step(self, features: list[torch.Tensor]) -> list[int]:
        if len(features) != self.num_hidden_layers:
            raise ValueError("Feature list does not match hidden layers")
        if self.replacement_rate == 0:
            return [0] * self.num_hidden_layers
        self.steps += 1
        step_counts = [0] * self.num_hidden_layers
        for i in range(self.num_hidden_layers):
            self.ages[i].add_(1.0)
            self._update_utility(i, features[i])
            eligible = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible.numel() == 0:
                continue
            self.accumulated[i] += self.per_step_replacement[i]
            count = int(self.accumulated[i])
            self.accumulated[i] -= count
            if count == 0:
                continue
            if count > eligible.numel():
                raise RuntimeError("Replacement count exceeds eligible units")
            selected = eligible[torch.topk(-self.bias_corrected_util[i][eligible], count).indices]
            outgoing = self._outgoing_indices(i, selected)
            current, following = self.layers[i], self.layers[i + 1]
            replacement = torch.empty(
                [count] + list(current.weight.shape[1:]), device="cpu"
            ).uniform_(-self.bounds[i], self.bounds[i], generator=self.rng).to(self.device)
            current.weight[selected] = replacement
            if current.bias is not None:
                current.bias[selected] = 0.0
            following.weight[:, outgoing] = 0.0
            self.ages[i][selected] = 0.0
            self.util[i][selected] = 0.0
            self.mean_feature_act[i][selected] = 0.0
            self.mean_abs_feature_act[i][selected] = 0.0
            self.replacements[i] += count
            step_counts[i] = count
            self.replacement_events.append({
                "step": self.steps,
                "layer": i,
                "indices": [int(x) for x in selected.cpu()],
            })
        return step_counts

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "replacements_by_layer": list(self.replacements),
            "replacement_events": list(self.replacement_events),
            "accumulated_fraction_by_layer": list(self.accumulated),
            "age_ranges_by_layer": [[float(x.min()), float(x.max())] for x in self.ages],
        }
