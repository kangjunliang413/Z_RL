# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from z_rl.modules import EmpiricalNormalization, MLP
from z_rl.models.composition import ComposableModel, ObsLatentAdapter, LatentSpec
from z_rl.utils import resolve_obs_temporal_selector


@dataclass
class MLPEncoderLatentSpec(LatentSpec):
    """Encode the single active ``policy`` observation group with an MLP."""

    encoder_latent_dim: int = 128
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256,)
    encoder_activation: str = "elu"
    concat_last_obs: bool = False

    def validate(self, model: nn.Module) -> None:
        """Require exactly one active observation group named ``policy``."""
        if getattr(model, "obs_groups", None) != ["policy"]:
            raise ValueError(
                "`MLPEncoderLatentSpec` requires exactly one active observation group named 'policy'. "
                f"Got {getattr(model, 'obs_groups', None)}."
            )

    def build(self, model: nn.Module) -> nn.Module:
        append_obs = None
        # whether to append the last observation of the policy group
        if self.concat_last_obs:
            append_obs = resolve_obs_temporal_selector("policy", "last", model.obs_group_time_slice_map)
        # build obs normalizers
        if model.obs_normalization is False:
            obs_normalizer: nn.Module = nn.Identity()
        else:
            normalization_cfg = {} if model.obs_normalization is True else model.obs_normalization
            obs_normalizer = EmpiricalNormalization(model.obs_dim, **normalization_cfg)
        # return obs adapter
        return ObsLatentAdapter(
            obs_groups=model.obs_groups,
            obs_normalizer=obs_normalizer,
            encoder=MLP(model.obs_dim, self.encoder_latent_dim, self.encoder_hidden_dims, self.encoder_activation),
            append_obs=append_obs,
        )

    def get_latent_dim(self, model: nn.Module) -> int:
        if not self.concat_last_obs:
            return self.encoder_latent_dim
        return self.encoder_latent_dim + resolve_obs_temporal_selector(
            "policy", "last", model.obs_group_time_slice_map
        ).dim


class MLPEncoderModel(ComposableModel):
    """Named preset that installs ``MLPEncoderLatentSpec``."""

    latent_spec_class = MLPEncoderLatentSpec
