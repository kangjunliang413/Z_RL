# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for composite optimizer utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

from z_rl.modules import MLP
from z_rl.utils.opt import MuonAdamWWrapper, OptimizerGroup


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = MLP(4, 2, [8, 8], activation="elu", first_non_muon=True, last_non_muon=True)


def test_mlp_marks_first_and_last_linear_as_non_muon() -> None:
    mlp = MLP(4, 2, [8, 8], activation="elu", first_non_muon=True, last_non_muon=True)
    linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]

    assert getattr(linear_layers[0].weight, "_non_muon", False)
    assert getattr(linear_layers[-1].weight, "_non_muon", False)
    assert not getattr(linear_layers[1].weight, "_non_muon", False)


def test_optimizer_group_steps_all_wrapped_optimizers() -> None:
    model = nn.Linear(2, 2)
    opt_a = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model.parameters(), lr=0.2)
    group = OptimizerGroup([opt_a, opt_b])

    loss = model(torch.ones(1, 2)).sum()
    group.zero_grad()
    loss.backward()
    group.step()

    assert len(group.state_dict()["optimizers"]) == 2


def test_muon_adamw_wrapper_splits_parameters() -> None:
    if not hasattr(torch.optim, "Muon"):
        return

    model = _TinyModel()
    wrapper = MuonAdamWWrapper([model], lr=1e-3)
    state = wrapper.state_dict()

    assert len(state["optimizers"]) == 2
