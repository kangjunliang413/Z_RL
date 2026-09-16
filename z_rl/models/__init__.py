# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .composition import ComposableModel, GroupObsLatentAdapter, ObsLatentAdapter, HeadSpec, LatentSpec
from .mlp_model import MLPModel
from .variants import (
    EncoderMLPModel,
    MLPEncoderLatentSpec,
    MoEHeadSpec,
    MoEModel,
)
from .rnn_model import RNNModel
from .cnn_model import CNNModel


__all__ = [
    "MLPModel",
    "ComposableModel",
    "ObsLatentAdapter",
    "GroupObsLatentAdapter",
    "LatentSpec",
    "HeadSpec",
    "EncoderMLPModel",
    "MLPEncoderLatentSpec",
    "MoEHeadSpec",
    "MoEModel",
    "RNNModel",
    "CNNModel",
]
