# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MoEModel resume and ONNX export."""

from __future__ import annotations

import tempfile

import onnx
import pytest
import torch

from tests.conftest import make_obs
from z_rl.models import MoEModel

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_moe_model(stochastic: bool = True) -> tuple[MoEModel, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    distribution_cfg = None
    if stochastic:
        distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    model = MoEModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS if stochastic else 1,
        activation="elu",
        distribution_cfg=distribution_cfg,
        num_experts=4,
        expert_hidden_dims=[16],
    )
    return model, obs


class TestMoEModelResumeAndExport:
    def test_load_state_dict_restores_deterministic_output(self) -> None:
        source, obs = _make_moe_model()
        target, _ = _make_moe_model()
        target.load_state_dict(source.state_dict())
        source.eval()
        target.eval()
        assert torch.allclose(source(obs), target(obs), atol=1e-6)

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_after_training_forward(self) -> None:
        """Export must succeed even when the router cache still holds a training graph."""
        actor, obs = _make_moe_model()
        actor(obs)
        actor.eval()

        onnx_model = actor.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
            assert [i.name for i in loaded.graph.input] == ["obs"]
            assert [o.name for o in loaded.graph.output] == ["actions"]
