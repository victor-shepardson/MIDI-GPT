"""controls["velocity_mask"]: resolving it to per-level weights, validating it, and applying it to a distribution."""

import pytest
import torch

from midigpt.inference.config import (
    resolve_velocity_allow_set,
    validate_velocity_mask,
    velocity_level_of,
    velocity_level_weights,
)
from midigpt.inference.session import SamplingSession


def test_level_of_matches_quantizer_ends():
    assert velocity_level_of(0, 32) == 0
    assert velocity_level_of(1, 32) == 1
    assert velocity_level_of(127, 32) == 31
    assert velocity_level_of(64, 128) == 64


def test_allow_set_from_min_max_and_list():
    assert resolve_velocity_allow_set({"min": 60, "max": 62}) == {60, 61, 62}
    assert resolve_velocity_allow_set({"velocities": [10, 200]}) == {10}
    assert resolve_velocity_allow_set({"shape": {"type": "normal", "mean": 64, "std": 8}}) is None


def test_hard_weights_128_levels():
    w = velocity_level_weights({"min": 60, "max": 70}, 128)
    assert w[0] == 0.0
    assert all(w[v] == 1.0 for v in range(60, 71))
    assert sum(w) == 11


def test_hard_weights_32_levels_partial_bin():
    w = velocity_level_weights({"min": 64, "max": 127}, 32)
    # a bin straddling 64 gets the share of its velocities that are allowed
    straddle = velocity_level_of(64, 32)
    assert 0.0 < w[straddle] <= 1.0
    assert w[velocity_level_of(40, 32)] == 0.0
    assert w[31] == 1.0


def test_normal_shape_peaks_at_mean():
    w = velocity_level_weights({"shape": {"type": "normal", "mean": 90, "std": 5}}, 128)
    assert max(range(128), key=lambda i: w[i]) == 90


@pytest.mark.parametrize("bad", [
    {"foo": 1},
    {"velocities": [0]},
    {"velocities": [64], "min": 1},
    {"min": 100, "max": 50},
    {"shape": {"type": "triangle"}},
    {"min": 10, "max": 20, "shape": {"type": "uniform", "min": 100, "max": 120}},
])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate_velocity_mask(bad)


def test_validate_accepts():
    validate_velocity_mask({"min": 40, "max": 90, "shape": {"type": "normal", "mean": 70, "std": 10}})


def test_apply_keeps_velocity_mass_and_zeroes_disallowed():
    probs = torch.tensor([0.5, 0.1, 0.2, 0.2])  # token 0 is not a velocity; 1-3 are levels 0-2
    weights = torch.tensor([0.0, 1.0, 0.0])
    out = SamplingSession._apply_velocity_mask(probs, (1, 4), weights)
    assert torch.allclose(out, torch.tensor([0.5, 0.0, 0.5, 0.0]))


def test_apply_falls_back_to_weights_when_model_gives_allowed_zero():
    probs = torch.tensor([0.4, 0.6, 0.0, 0.0])
    weights = torch.tensor([0.0, 1.0, 3.0])
    out = SamplingSession._apply_velocity_mask(probs, (1, 4), weights)
    assert torch.allclose(out, torch.tensor([0.4, 0.0, 0.15, 0.45]))


def test_apply_leaves_probs_without_velocity_mass():
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = SamplingSession._apply_velocity_mask(probs, (1, 4), torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(out, probs)
