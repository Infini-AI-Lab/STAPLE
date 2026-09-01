# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict adapter for Engram-OLMo2.

The backbone shares OLMo2's HF<->torchtitan mapping verbatim, so a vanilla
OLMo2 checkpoint loads into the backbone unchanged (Engram only *adds*
parameters). Engram parameters (``layers.{}.engram.*``) have no HF counterpart;
they are left untouched here and initialized fresh when not present in a loaded
checkpoint.
"""

from __future__ import annotations

from typing import Any

from torchtitan.models.olmo2.state_dict_adapter import OLMo2StateDictAdapter

__all__ = ["EngramOLMo2StateDictAdapter"]


class EngramOLMo2StateDictAdapter(OLMo2StateDictAdapter):
    """OLMo2 adapter; passes Engram-only params through untouched.

    ``to_hf`` drops Engram parameters (no HF equivalent); ``from_hf`` never
    produces them, so they keep their freshly initialized values.
    """

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        backbone = {k: v for k, v in state_dict.items() if ".engram." not in k}
        return super().to_hf(backbone)
