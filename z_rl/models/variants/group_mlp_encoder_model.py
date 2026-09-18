from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from z_rl.models.composition import ComposableModel, GroupObsLatentAdapter, LatentSpec
from z_rl.modules import EmpiricalNormalization, MLP


@dataclass
class GroupMLPLatentSpec(LatentSpec):
    """Encode each active 1D observation group with its own MLP."""

    encoder_cfgs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _latent_dim: int = field(init=False, repr=False, default=0)

    def validate(self, model: nn.Module) -> None:
        expected = set(model.obs_groups)
        configured = set(self.encoder_cfgs)
        if configured != expected:
            raise ValueError(
                "`GroupMLPLatentSpec.encoder_cfgs` must declare every active observation group, "
                f"got {sorted(configured)} vs obs_groups={sorted(expected)}."
            )
        for group in model.obs_groups:
            if len(model.obs_group_shapes[group]) != 1:
                raise ValueError(
                    f"`GroupMLPLatentSpec` only supports 1D observations, got "
                    f"shape {model.obs_group_shapes[group]} for '{group}'."
                )
            if "output_dim" not in self.encoder_cfgs[group]:
                raise ValueError(f"Encoder config for '{group}' requires `output_dim`.")

    def build(self, model: nn.Module) -> nn.Module:
        encoders = {}
        normalizers = {}
        group_dims = []
        self._latent_dim = 0

        normalization_cfg = None
        if model.obs_normalization is not False:
            normalization_cfg = {} if model.obs_normalization is True else model.obs_normalization

        for group in model.obs_groups:
            input_dim = int(model.obs_group_shapes[group][-1])
            cfg = dict(self.encoder_cfgs[group])
            output_dim = int(cfg.pop("output_dim"))
            hidden_dims = cfg.pop("hidden_dims", ())
            activation = cfg.pop("activation", "swish")
            if cfg:
                raise ValueError(f"Unsupported encoder configuration keys for '{group}': {list(cfg)}")
            if hidden_dims:
                encoder = MLP(
                    input_dim,
                    output_dim,
                    hidden_dims,
                    activation,
                    first_non_muon=True,
                    last_non_muon=True,
                )
            else:
                encoder = nn.Linear(input_dim, output_dim)
                encoder.weight._non_muon = True
            encoders[group] = encoder
            normalizers[group] = (
                nn.Identity()
                if normalization_cfg is None
                else EmpiricalNormalization(input_dim, **normalization_cfg)
            )
            group_dims.append(input_dim)
            self._latent_dim += output_dim

        return GroupObsLatentAdapter(
            obs_groups=list(model.obs_groups),
            obs_group_dims=group_dims,
            encoders=encoders,
            obs_normalizers=normalizers,
        )

    def get_latent_dim(self, model: nn.Module) -> int:
        del model
        return self._latent_dim


class GroupMLPEncoderModel(ComposableModel):
    """Named preset that installs ``GroupMLPLatentSpec``."""

    latent_spec_class = GroupMLPLatentSpec
