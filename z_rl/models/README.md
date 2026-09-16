# Models

This directory contains the model implementations used by Z-RL and the explicit composition API for latent/head customization.

## Overview

Main models:

- `MLPModel`: base model for vector observations.
- `RNNModel`: recurrent model built on top of the MLP pipeline.
- `CNNModel`: model for mixed 1D/2D observations.
- `ComposableModel`: thin `MLPModel` wrapper that accepts `latent_spec` and `head_spec`.

Predefined variants live in [`variants/`](https://github.com/syw-robotics/z_rl/tree/main/z_rl/models/variants):

- `EncoderMLPModel`: `ComposableModel` variant with `MLPEncoderLatentSpec` as the latent stage.
- `MoEModel`: `ComposableModel` variant whose head is replaced by a Mixture-of-Experts module. MoE head shape is controlled by `expert_hidden_dims` and `gate_hidden_dims`. This MoE implementation suppors the experts run **in parallel**. Pair it with `MoEPPO` to add expert-balance (and optional gate-entropy) routing regularizers.

## Export Logic

Z-RL keeps policy export ONNX-only. `MLPModel` export follows the runtime structure while using a tensor-only adapter:

```text
flat ONNX input -> latent_adapter.as_export_module() -> head -> deterministic_output
```

Export entry points:

- `model.as_onnx(...)`

Runtime latent adapters consume the structured observation `TensorDict`. If a custom adapter is not directly compatible
with the flat tensor passed by ONNX export, implement `as_export_module()` on the adapter and return a tensor-only module.

## Composition API

Preferred customization lives in [`composition/`](/home/syw/.gitrepos/z_rl/tree/main/z_rl/models/composition):

- `composition/specs.py`: base classes `LatentSpec` and `HeadSpec`
- `composition/composable_model.py`: `ComposableModel`
- `composition/adapters.py`: `ObsLatentAdapter` (concat groups, one normalizer, one encoder) and
  `GroupObsLatentAdapter` (per-group normalizer and encoder, then concat)
- `variants/`: named presets and their variant-specific latent/head specs

The user-facing unit is the spec. Config can point `ComposableModel` at a spec without a named model subclass:

```python
actor = {
    "class_name": "ComposableModel",
    "latent_spec": {"class_name": "MLPEncoderLatentSpec", "encoder_latent_dim": 128},
    "head_spec": {"class_name": "MoEHeadSpec", "num_experts": 4, "expert_hidden_dims": [256]},
}
```

Programmatic construction still accepts instances, classes, or the same config dicts:

```python
from z_rl.models import ComposableModel, MLPEncoderLatentSpec, MoEHeadSpec

model = ComposableModel(
    ...,
    latent_spec=MLPEncoderLatentSpec(encoder_latent_dim=128),
    head_spec=MoEHeadSpec(num_experts=4),
)
```

`LatentSpec` requires `build(model)` and `get_latent_dim(model)`. Optional:

- `validate(model)`: no-op by default

The adapter returned by `build(model)` should implement `forward(obs: TensorDict)`. It may also implement
`update_normalization(obs)` when it owns normalization statistics. Built-in adapters implement `as_export_module()` so
training `forward` stays TensorDict-only.

Omit `latent_spec` to keep `model.obs_dim`.

`HeadSpec` requires `build(model, input_dim, output_dim, activation)`. `validate(model)` is a no-op by default.

If a latent spec changes the latent width, `ComposableModel` rebuilds the head with the new dimension.


Variant-owned specs:

- `models/variants/encoder_mlp_model.py`: `MLPEncoderLatentSpec` and `EncoderMLPModel`
- `models/variants/moe_model.py`: `MoEHeadSpec` and `MoEModel`

When a latent or head spec only serves one concrete model variant, keep that spec in the same variant module rather
than under `composition/`.

## Maintenance Notes

- Keep runtime and export structure aligned.
- Prefer adapter modules over ad-hoc `forward()` logic.
- Keep normalization owned by the adapter that consumes the corresponding observations.
- When changing composition contracts, update this README and `z_rl/cli/plugin_templates/models.py`.
