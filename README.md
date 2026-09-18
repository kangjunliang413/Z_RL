# Z-RL

**Z-RL** is a lightweight reinforcement learning infrastructure derived from [RSL-RL](https://github.com/leggedrobotics/rsl_rl), redesigned for faster iteration in robotics projects. Environment adaptors currently support IsaacLab manager-based RL environments and Active Adaptation environments.

`Z` is the last letter of the alphabet, we hope Z-RL is your **last** time dedicatedly reviewing RL infra code.  

## 🎯 Key Features

Compared with plain RSL-RL style usage, Z-RL emphasizes:

- **Composable design**: for quick implementation of customized algorithms and models.
- **Plugin system**: so project-specific logic can live outside the core library.
- **Adaptor layer**: for different rl environment integration, currently including IsaacLab `ManagerBasedRLEnv` support.
- **`ObsSelector` utility**: cached, reusable observation selectors that make observation operations safe and efficeient.

## 📦 Installation

Before installing Z-RL, make sure Python `3.9+` is available.

It is recommended to use a virtual environment (`venv`, `conda`, or `uv`) and activate it first.

```bash
git clone https://github.com/syw-robotics/z_rl
cd z_rl
python -m pip install -e .

# [optional] update torch version to support muon optimizer
python -m pip install -e ".[muon]" 
```

<!-- ### Muon optimizer (optional) -->
<!--  -->
<!-- PPO supports an optional `use_muon` flag. When enabled, hidden 2-D weight matrices are optimized with -->
<!-- [`torch.optim.Muon`](https://pytorch.org/docs/stable/generated/torch.optim.Muon.html); biases, normalization -->
<!-- parameters, policy std heads, and input/output linear layers stay on AdamW. This requires PyTorch 2.9+. -->
<!--  -->
<!-- Install the optional extra (or ensure your environment already satisfies `torch>=2.9`): -->
<!--  -->
<!-- ```bash -->
<!-- python -m pip install -e ".[muon]" -->
<!-- ``` -->
<!--  -->
<!-- Enable it in your algorithm config: -->
<!--  -->
<!-- ```python -->
<!-- train_cfg["algorithm"]["use_muon"] = True -->
<!-- ``` -->
<!--  -->
<!-- For Isaac Lab configs, set `use_muon=True` on `ZRlPpoAlgorithmCfg`. When `use_muon` is enabled, the `optimizer` -->
<!-- field is ignored. Optimizer state in checkpoints is not compatible with plain Adam/AdamW runs, so resume training -->
<!-- with `load_cfg["optimizer"]=False` when switching optimizer modes. -->

## 🚀 Usage

For detailed module guides, see:

- [IsaacLab adaptor README](z_rl/adaptor/isaaclab/README.md)
- [Active Adaptation adaptor README](z_rl/adaptor/active_adaptation/README.md)
- [Algorithms README](z_rl/algorithms/README.md)
- [Models README](z_rl/models/README.md)

## 🛠️ CLI Tools

After installing Z-RL, the following command line tools are available:

### `z-rl-plugin-init`

Generate a minimal external plugin package scaffold:

```bash
z-rl-plugin-init

# z-rl-plugin-init --path ./my_zrl_plugin --name z_rl_plugin_example
```

### `z-rl-checkpoint-editor`

Launch a Gradio UI for inspecting all nested PyTorch checkpoint contents with path filtering and renaming top-level
checkpoint keys. This is useful when adapting older checkpoints, for example renaming `student_state_dict` to
`actor_state_dict`.

```bash
z-rl-checkpoint-editor
```

## 🔌 Plugin System

Z-RL supports **external plugin packages** so your custom algorithms/models/modules stay isolated from upstream core code.

### Generate a plugin template

After installing Z-RL, run:

```bash
z-rl-plugin-init

# z-rl-plugin-init --path ./my_zrl_plugin --name z_rl_plugin_example
```

This creates a minimal package scaffold containing:

- a custom `PPOLossSpec` example
- custom `LatentSpec` / `HeadSpec` examples
- plugin-side IsaacLab config classes that point `ComposablePPO` / `ComposableModel` at those specs (`rl_cfg.py`)

```
.
├── pyproject.toml
├── README.md
└── z_rl_plugin_example
    ├── algorithms
    │   ├── __init__.py
    │   └── my_loss.py
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   └── my_model.py
    ├── modules
    │   └── __init__.py
    └── rl_cfg.py

```

Implement your specs, then install your plugin in editable mode:

```bash
cd my_zrl_plugin
python -m pip install -e .
```

## 📋 TODOs

- ✅ Added and tested `MoEModel` as a model extension example
- ✅ Added and tested `EncoderEstimationPPO` as a PPO extension example
- ✅ Keep deployment/export ONNX-only and remove TorchScript export paths
- ✅ Clarify latent adapter runtime/export contracts in model docs and plugin templates
- ✅ Reorganize `CNNModel` into `CNNLatentSpec` (single 2D observation group)
- ✅ Make `RNNModel` a composable recurrent backbone (`CNNLatentSpec` for CNN+RNN)
- ❌ Support multiple PPO loss specs for combining auxiliary objectives
- ❌ Add ONNX deployment notes for MLP/RNN/CNN policies
- ❌ Update tests after the adapter/export contract settles


## 📄 License

BSD-3-Clause. See [LICENSE](LICENSE).
