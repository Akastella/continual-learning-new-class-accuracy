import copy

import torch

from experiments.r18_cbp_official.cbp import OfficialConvGnT
from experiments.r18_cbp_official.engine import Network
from experiments.r9_followup.engine import Network as PrimaryNetwork


def test_architecture_and_initialization_are_primary_exact():
    with torch.random.fork_rng():
        torch.manual_seed(10201)
        primary = PrimaryNetwork("split_cifar10_classil", 256)
    with torch.random.fork_rng():
        torch.manual_seed(10201)
        cbp = Network("split_cifar10_classil", 256)
    assert primary.state_dict().keys() == cbp.state_dict().keys()
    assert all(torch.equal(primary.state_dict()[k], cbp.state_dict()[k]) for k in primary.state_dict())


def test_zero_rate_step_changes_no_parameters():
    torch.manual_seed(9)
    model = Network("split_cifar10_classil", 256)
    before = copy.deepcopy(model.state_dict())
    x = torch.randn(2, 3, 32, 32)
    _, features = model.forward_with_features(x)
    gnt = OfficialConvGnT(model.weight_layers(), replacement_rate=0.0, decay_rate=0.99,
                          maturity_threshold=0, num_last_filter_outputs=16,
                          rng_seed=7, device=torch.device("cpu"))
    ages = [x.clone() for x in gnt.ages]
    utilities = [x.clone() for x in gnt.util]
    assert gnt.step(features) == [0, 0, 0, 0]
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
    assert gnt.steps == 0
    assert all(torch.equal(before_age, after_age) for before_age, after_age in zip(ages, gnt.ages))
    assert all(torch.equal(before_u, after_u) for before_u, after_u in zip(utilities, gnt.util))


def test_primary_feature_mapping_and_stateless_sgd():
    torch.manual_seed(11)
    model = Network("split_cifar10_classil", 256)
    logits, features = model.forward_with_features(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert [tuple(x.shape) for x in features] == [
        (2, 32, 16, 16), (2, 64, 8, 8), (2, 2048), (2, 256)
    ]
    gnt = OfficialConvGnT(model.weight_layers(), replacement_rate=0.0, decay_rate=0.99,
                          maturity_threshold=1000, num_last_filter_outputs=16,
                          rng_seed=7, device=torch.device("cpu"))
    outgoing = gnt._outgoing_indices(2, torch.tensor([0, 2]))
    assert torch.equal(outgoing, torch.tensor([*range(16), *range(32, 48)]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    assert optimizer.state == {}


def test_forced_replacement_resets_hidden_to_head_connection():
    torch.manual_seed(9)
    model = Network("split_cifar10_classil", 256)
    x = torch.randn(2, 3, 32, 32)
    _, features = model.forward_with_features(x)
    gnt = OfficialConvGnT(model.weight_layers(), replacement_rate=1.0, decay_rate=0.9,
                          maturity_threshold=0, num_last_filter_outputs=16,
                          rng_seed=7, device=torch.device("cpu"))
    counts = gnt.step(features)
    assert counts == [32, 64, 128, 256]
    assert torch.count_nonzero(model.head.weight) == 0
    assert gnt.replacements == counts
