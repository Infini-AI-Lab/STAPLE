# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engram conditional-memory module on top of the OLMo2 backbone.

Engram (DeepSeek, "Conditional Memory via Scalable Lookup") augments a
Transformer backbone with large, hash-addressed N-gram embedding tables that
are retrieved via O(1) lookups and fused into the residual stream through a
context-aware gate. This experiment injects Engram modules into specific
OLMo2 layers; the embedding tables are row/vocab-parallel sharded across the
data-parallel group (FSDP is not used for them).

This is an ``experiments/`` package: it reuses core OLMo2 / common components
but does not modify them.
"""
