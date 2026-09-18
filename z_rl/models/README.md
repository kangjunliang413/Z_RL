# Models

Runtime: `obs TensorDict -> latent adapter -> (RNN) -> head -> (distribution) -> output`.

- `mlp_model.py`: `MLPModel` — concat 1D groups, optional normalize, MLP head.
- `composition/`: `LatentSpec` / `HeadSpec`, `ComposableModel`, adapters.
- `variants/`: named presets. A spec that only serves one preset lives next to it. Specs are re-exported from `z_rl.models` so `class_name` lookup works; `variants/__init__.py` only exports model classes.

## Variants

| Preset | Role | Spec |
| --- | --- | --- |
| `RNNModel` | RNN between adapter and head | none; kwargs `rnn_type`, `rnn_hidden_dim`, `rnn_num_layers` |
| `MLPEncoderModel` | MLP encoder latent | `MLPEncoderLatentSpec` |
| `GroupMLPEncoderModel` | Per-group MLP latent | `GroupMLPLatentSpec` |
| `CNNModel` | CNN latent | `CNNLatentSpec` |
| `MoEModel` | MoE head | `MoEHeadSpec` |
| `SimBaModel` | SimBaV2 hyperspherical head | `SimBaHeadSpec` |

Preset kwargs that match spec fields are bound onto the spec. Specs also work on `ComposableModel` / `RNNModel` without the named preset.

### RNNModel

```text
obs groups -> adapter / latent_spec -> RNN -> head
```

RNN input width is the adapter output; the head consumes `rnn_hidden_dim`. ONNX is vector-obs only (`obs`, `h_in`, `c_in`) — extra tensors such as a CNN image are not packed.

### MLPEncoderModel

Requires a single `policy` group. Optional `concat_last_obs` appends the last policy frame (from `obs_group_time_slice_map`) after the encoder.

### CNNModel

One 2D group (`image_obs_group`) through a CNN; remaining groups must be 1D. Each group is normalized/encoded, then concatenated. No multi-2D groups, no actor/critic CNN sharing.

ONNX: 1D groups packed as `obs`, image stays at native rank.

`CNNLatentSpec`: `image_obs_group`, `cnn_cfg` (flattened CNN output required), `cnn_projection_cfg`, `concat_last_obs`.

### GroupMLPEncoderModel and SimBaModel

`GroupMLPLatentSpec` encodes every active 1D observation group independently. Each entry in `encoder_cfgs` requires
`output_dim` and optionally accepts `hidden_dims` and `activation`. Different groups may use different output widths.

`SimBaHeadSpec` replaces the ordinary MLP head with a SimBaV2 hyperspherical residual network. It accepts
`hidden_dim`, `num_blocks`, `expansion`, and `c_shift`. The specs can be composed without a dedicated actor-critic:

```python
model = ComposableModel(
    ...,
    latent_spec={
        "class_name": "GroupMLPLatentSpec",
        "encoder_cfgs": {
            "proprio": {"output_dim": 128, "hidden_dims": [256]},
            "object": {"output_dim": 64, "hidden_dims": [128]},
        },
    },
    head_spec={"class_name": "SimBaHeadSpec", "hidden_dim": 512, "num_blocks": 2, "expansion": 4},
)
```

### MoEModel

Dense MoE: every expert runs, the gate mixes. Compute scales with `num_experts`. Pair with `MoEPPO`.

`MoEHeadSpec`: `num_experts`, `expert_hidden_dims`, `gate_hidden_dims` (`None` = linear gate).

`MoERoutingLossSpec`: `expert_balance_loss` (KL to uniform, default `1e-4`), `gate_entropy_loss` (default `0`). At least one of actor/critic must have an MoE head.

Pretrained experts load into `head.experts` only, never the gate. Use `pretrained_expert_path` **or** `pretrained_expert_specs`. Checkpoints may be a raw `state_dict` or wrapped under `actor_state_dict` / `state_dict`. MoE keys (`head.experts.weights.0`) and MLP heads (`head.0.weight`) are both accepted; MLP weights are transposed into the stacked `[E, in, out]` layout.

## Specs

```python
from z_rl.models import ComposableModel, MLPEncoderLatentSpec, MoEHeadSpec

model = ComposableModel(
    ...,
    latent_spec=MLPEncoderLatentSpec(encoder_latent_dim=128),
    head_spec=MoEHeadSpec(num_experts=4),
)
```

Config form: `{"class_name": "MLPEncoderLatentSpec", ...}`. `RNNModel` can take `CNNLatentSpec` / `MoEHeadSpec` the same way.

- `LatentSpec`: `build(model)`, `get_latent_dim(model)`; optional `validate`. Omit to keep `model.obs_dim`.
- `HeadSpec`: `build(model, input_dim, output_dim, activation)`.
- `ObsLatentAdapter`: concat → normalize → encode. `GroupObsLatentAdapter`: per-group then concat. Runtime `forward(obs: TensorDict)`. Implement `as_export_module()` for ONNX.

## Export

Entry point: `model.as_onnx(...)`. Training adapters take a `TensorDict`; export uses `as_export_module()` on a flat tensor. Non-vector ONNX inputs also need `export_dummy_inputs()` and `export_input_names()`.

## Maintenance

Keep runtime and export aligned. Normalization stays on the adapter that consumes those observations. Contract changes: this README and `z_rl/cli/plugin_templates/models.py`.
