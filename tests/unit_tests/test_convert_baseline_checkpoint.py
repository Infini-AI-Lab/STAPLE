# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from scripts.checkpoint_conversion.convert_baseline_to_torchtitan import (
    copy_hf_assets,
    expected_model_keys,
    validate_converted_model_state,
    validate_dcp_output,
)


def test_validate_converted_model_state_requires_complete_namespace():
    model_keys = expected_model_keys(n_layers=2)
    model_state = {key: torch.empty(0) for key in model_keys}
    validate_converted_model_state(model_state, n_layers=2)

    model_state.pop("layers.0.attention.qkv_linear.wq.weight")
    with pytest.raises(ValueError, match="qkv_linear.wq.weight"):
        validate_converted_model_state(model_state, n_layers=2)


def test_validate_dcp_output_requires_model_keys_and_optimizer():
    metadata = SimpleNamespace(
        state_dict_metadata={
            "layers.0.attention.qkv_linear.wq.weight": object(),
            "optimizer.state.layers.0.attention.qkv_linear.wq.weight.step": object(),
        }
    )
    with mock.patch(
        "scripts.checkpoint_conversion.convert_baseline_to_torchtitan.dcp.FileSystemReader"
    ) as reader:
        reader.return_value.read_metadata.return_value = metadata
        validate_dcp_output(
            Path("/checkpoint"),
            {"layers.0.attention.qkv_linear.wq.weight"},
            expect_optimizer=True,
        )

        with pytest.raises(RuntimeError, match="missing 1 model keys"):
            validate_dcp_output(
                Path("/checkpoint"),
                {
                    "layers.0.attention.qkv_linear.wq.weight",
                    "layers.0.attention.qkv_linear.wk.weight",
                },
                expect_optimizer=True,
            )


def test_copy_hf_assets_does_not_overwrite_dcp_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        (source / "config.json").write_text('{"model_type": "olmo2"}')
        (source / ".metadata").write_text("source metadata")
        (output / ".metadata").write_text("converted metadata")

        copy_hf_assets(source, output)

        assert (output / "config.json").read_text() == '{"model_type": "olmo2"}'
        assert (output / ".metadata").read_text() == "converted metadata"
