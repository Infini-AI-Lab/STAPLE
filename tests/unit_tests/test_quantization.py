# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.experiments.olmo2_moe_stem_dag.moe import StaticStemGroupedExperts
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.nn_modules import Linear


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "olmo2", "--config", "olmo2_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama4", "--config", "llama4_debugmodel_fp8"]
    )
    model_config = config.model_spec.model
    # ``llama4_debugmodel_fp8`` declares a Float8GroupedExpertsConverter, which
    # model_registry applies while building the spec.
    assert has_quantization(model_config)


def test_quantized_grouped_experts():
    """Quantized GroupedExperts: _owner, subclass handling, extra config fields."""
    # Base case
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (StaticStemGroupedExperts has extra granularity field)
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(StaticStemGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(StaticStemGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, StaticStemGroupedExperts)
    assert issubclass(float8_cls, StaticStemGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "granularity")
    assert hasattr(float8_cls.Config, "granularity")
