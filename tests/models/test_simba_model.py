"""Tests for grouped MLP observation encoding and SimBaV2 heads."""

from __future__ import annotations

import tempfile

import onnx
import pytest
import torch
from tensordict import TensorDict

from z_rl.models import ComposableModel, GroupMLPEncoderModel, SimBaModel
from z_rl.modules import SimBa

NUM_ENVS = 4


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "proprio": torch.randn(NUM_ENVS, 6),
            "object": torch.randn(NUM_ENVS, 4),
        },
        batch_size=[NUM_ENVS],
    )


ENCODER_CFGS = {
    "proprio": {"output_dim": 5, "hidden_dims": [8]},
    "object": {"output_dim": 3, "hidden_dims": []},
}


class TestGroupMLPLatentSpec:
    def test_uses_independent_group_widths(self) -> None:
        obs = _make_obs()
        model = GroupMLPEncoderModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            2,
            hidden_dims=[8],
            encoder_cfgs=ENCODER_CFGS,
            obs_normalization=True,
        )

        latent = model.get_latent(obs)
        packed = torch.cat((obs["proprio"], obs["object"]), dim=-1)

        assert latent.shape == (NUM_ENVS, 8)
        assert torch.allclose(latent, model.latent_adapter.as_export_module()(packed))
        assert set(model.latent_adapter.obs_normalizers) == {"proprio", "object"}

    def test_requires_every_active_group(self) -> None:
        obs = _make_obs()
        with pytest.raises(ValueError, match="must declare every active observation group"):
            GroupMLPEncoderModel(
                obs,
                {"actor": ["proprio", "object"]},
                "actor",
                2,
                hidden_dims=[8],
                encoder_cfgs={"proprio": ENCODER_CFGS["proprio"]},
            )


class TestSimBaHeadSpec:
    def test_composes_with_group_encoder(self) -> None:
        obs = _make_obs()
        model = ComposableModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            3,
            hidden_dims=[8],
            latent_spec={"class_name": "GroupMLPLatentSpec", "encoder_cfgs": ENCODER_CFGS},
            head_spec={"class_name": "SimBaHeadSpec", "hidden_dim": 16, "num_blocks": 2, "expansion": 2},
        )

        output = model(obs)
        output.square().mean().backward()

        assert isinstance(model.head, SimBa)
        assert output.shape == (NUM_ENVS, 3)
        assert torch.isfinite(output).all()
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    def test_named_preset_uses_simba_head(self) -> None:
        obs = _make_obs()
        model = SimBaModel(
            obs,
            {"critic": ["proprio", "object"]},
            "critic",
            1,
            hidden_dims=[8],
            hidden_dim=16,
            num_blocks=1,
            expansion=2,
        )

        assert isinstance(model.head, SimBa)
        assert model(obs).shape == (NUM_ENVS, 1)

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning:torch.onnx")
    def test_onnx_export(self) -> None:
        obs = _make_obs()[:1]
        model = ComposableModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            3,
            hidden_dims=[8],
            latent_spec={"class_name": "GroupMLPLatentSpec", "encoder_cfgs": ENCODER_CFGS},
            head_spec={"class_name": "SimBaHeadSpec", "hidden_dim": 8, "num_blocks": 1, "expansion": 2},
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                file.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            onnx.checker.check_model(onnx.load(file.name))
