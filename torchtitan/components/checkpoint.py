# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import functools
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast, Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import DefaultLoadPlanner, HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class _SelectiveMissingKeysLoadPlanner(DefaultLoadPlanner):
    """Allow only explicitly authorized keys to be absent from a checkpoint."""

    def __init__(self, allowed_missing_key_prefixes: tuple[str, ...]) -> None:
        super().__init__(allow_partial_load=True)
        self.allowed_missing_key_prefixes = allowed_missing_key_prefixes

    def create_local_plan(self):
        plan = super().create_local_plan()
        assert self.metadata is not None
        checkpoint_keys = set(self.metadata.state_dict_metadata)
        unauthorized_missing_keys = sorted(
            key
            for key in set(self.state_dict) - checkpoint_keys
            if not key.startswith(self.allowed_missing_key_prefixes)
        )
        if unauthorized_missing_keys:
            raise RuntimeError(
                f"Missing key in checkpoint state_dict: {unauthorized_missing_keys[0]}."
            )
        return plan


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


def _run_aws(args: list[str]) -> subprocess.CompletedProcess:
    """Run an ``aws`` CLI command, capturing output.

    Raises CalledProcessError on non-zero exit so the caller can log it.
    """
    return subprocess.run(args, check=True, capture_output=True, text=True)


def s3_upload_thread(
    upload_queue: queue.Queue,
    remote_root: str,
    keep_latest_k: int,
    is_purge_leader: bool,
):
    """Thread to push freshly-saved checkpoints to an S3 bucket.

    This is only used when ``checkpoint.remote_sync_path`` is set. Each item on
    the queue is a (local_path, step) tuple: the local checkpoint folder for the
    current step is synced to ``{remote_root}/step-{step}/``. After a successful
    upload, the purge leader trims the remote checkpoints to the latest k.

    Mirrors ``purge_thread`` so the (potentially slow) ``aws s3`` calls never
    block the training loop.

    Args:
        upload_queue (queue.Queue): Receives (local_path, step) tuples to upload
            and a Terminate signal to stop.
        remote_root (str): The S3 prefix to sync checkpoints under, e.g.
            ``s3://my-bucket/path/checkpoint``.
        keep_latest_k (int): Keep only the latest k checkpoints remotely. If 0,
            keep all.
        is_purge_leader (bool): Whether this rank is responsible for purging
            stale remote checkpoints. Only one rank (global rank 0) should purge
            to avoid concurrent ``aws s3 rm`` races, while every node-local
            leader uploads its own shards.
    """
    remote_root = remote_root.rstrip("/")
    try:
        while True:
            item = upload_queue.get()
            if isinstance(item, Terminate):
                return
            assert isinstance(item, tuple)
            local_path, step = item
            remote_path = f"{remote_root}/step-{step}/"
            logger.info("Checkpointer is uploading %s to %s.", local_path, remote_path)
            begin = time.monotonic()
            try:
                _run_aws(["aws", "s3", "sync", local_path, remote_path])
            except subprocess.CalledProcessError as e:
                logger.error(
                    "Failed to upload %s to %s: %s",
                    local_path,
                    remote_path,
                    e.stderr,
                )
                continue
            logger.info(
                "Checkpointer uploaded %s to %s in %.2f seconds.",
                local_path,
                remote_path,
                time.monotonic() - begin,
            )

            if is_purge_leader and keep_latest_k > 0:
                _purge_stale_remote_checkpoints(remote_root, keep_latest_k)
    finally:
        logger.info("Destroying the s3 upload thread.")


def _purge_stale_remote_checkpoints(remote_root: str, keep_latest_k: int) -> None:
    """List remote checkpoints and delete all but the latest k.

    The bucket is the source of truth (rather than an in-memory list) so this
    stays correct across job restarts and preemption.
    """
    try:
        # `aws s3 ls <prefix>/` lists immediate children; step folders show up as
        # "PRE step-<n>/" lines.
        result = _run_aws(["aws", "s3", "ls", f"{remote_root}/"])
    except subprocess.CalledProcessError as e:
        logger.error(
            "Failed to list remote checkpoints under %s: %s", remote_root, e.stderr
        )
        return

    discovered_checkpoints = []
    for line in result.stdout.splitlines():
        match = re.search(r"step-(\d+)/?$", line.strip())
        if match:
            discovered_checkpoints.append((int(match.group(1)), match.group(0)))

    discovered_checkpoints.sort()
    to_delete = discovered_checkpoints[: -1 * keep_latest_k]

    for step, name in to_delete:
        remote_path = f"{remote_root}/step-{step}/"
        logger.info("Checkpointer is deleting remote %s.", remote_path)
        begin = time.monotonic()
        try:
            _run_aws(["aws", "s3", "rm", "--recursive", remote_path])
        except subprocess.CalledProcessError as e:
            logger.error("Failed to delete remote %s: %s", remote_path, e.stderr)
            continue
        logger.info(
            "Checkpointer deleted remote %s in %.2f seconds.",
            remote_path,
            time.monotonic() - begin,
        )


class CheckpointManager(Configurable):
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is guaranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with config.folder
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert model state
            dicts between native format and other formats.

    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to enable checkpoint"""

        folder: str = "checkpoint"
        """
        The folder to store the checkpoints.
        When enable is set to true, checkpoints will be in {--dump_folder}/{--checkpoint.folder}.
        """

        interval: int = 500
        """Checkpointing interval in steps."""

        initial_load_path: str | None = None
        """
        This option specifies the path to the initial checkpoint to load, which is
        particularly useful for resuming training from a previous run with a
        different output path or when loading a checkpoint from a pre-trained model.
        If the checkpoint folder for the current run is not empty,
        located at {--dump_folder}/{--checkpoint.folder}, this option will be ignored.
        This feature allows users to load an initial checkpoint from a different folder and
        continue training, saving new checkpoints to the specified folder without affecting
        the existing ones.

        Note that the path should contain the full path to the checkpoint folder,
        including the step number, if any; for example,
        "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
        """

        initial_load_model_only: bool = True
        """
        This option specifies if only the model should be loaded during the initial
        checkpoint load. The option is only used when `initial_load_path` is specified.
        If False, the checkpoint at `initial_load_path` is treated as a standard training
        checkpoint, including optimizer, lr scheduler, training states, etc.
        The default setting for this option is True. Note that you will have to use
        `--checkpoint.no_initial_load_model_only` to override the default setting.
        """

        initial_load_allowed_missing_model_key_prefixes: list[str] = field(
            default_factory=list
        )
        """
        Model key prefixes that may be absent from an initial DCP checkpoint.
        This is useful when the target model adds newly initialized parameters to
        a pretrained model. Missing model keys that do not match one of these
        prefixes are rejected. Resume loading uses a separate, optimizer-only
        migration allowlist.
        """

        resume_allowed_missing_optimizer_state_key_prefixes: list[str] = field(
            default_factory=list
        )
        """
        Optimizer-state key prefixes that may be absent when resuming a legacy
        DCP checkpoint. This is intended only for narrowly scoped migrations
        where the missing optimizer slots have deterministic initial values.
        Every other missing resume key remains an error.
        """

        initial_load_in_hf: bool = False
        """
        Enable the use of HuggingFace's safetensors format for checkpointing. The option
        is only used when `initial_load_path` is specified. This will load checkpoints
        in HF's model definition and safetensors format instead of the default torchtitan
        model definition and DCP format, after necessary model state dict transformation.
        `initial_load_model_only` must be true because safetensors doesn't support saving
        non-tensors. The default value is False.
        """

        initial_load_in_hf_quantized: bool = False
        """
        Enable loading of HuggingFace's safetensors format with quantized state dict keys. The option
        is only used when `initial_load_path` and `initial_load_path_in_hf` is specified. This will load
        checkpoints in HF's model definition and dequantize on model weights if necessary. To support
        this parameter, the model need to define proper HuggingFaceStorageReader to perform dequantize.
        """

        last_save_model_only: bool = True
        """
        When last_save_model_only=True, only the model will be saved at the end of training,
        the last save.  With this, checkpoints can be loaded using `torch.load(..., weights_only=True)`
        after conversion.  When last_save_model_only=False, the full checkpoint will be saved.
        A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
        The default value is True.
        """

        last_save_in_hf: bool = False
        """
        Enable the use of Hugging Face's safetensors format for checkpointing. This will save the
        final checkpoints in safetensors format instead of the default DCP format, after necessary
        model state dict transformation. There will be a performance cost in using this as we need
        to consolidate the sharded tensors to full tensors as a separate step.
        last_save_model_only must be true because safetensors doesn't support saving
        non-tensors. On load, this argument isn't needed as we will detect whether the loaded
        checkpoint is in safetensors format or not. The default value is False.
        """

        export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
        """
        Converts to the specified precision when training completes and last_save_model_only=true.
        """

        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """
        Which async checkpoint mode to use. Currently there are 3 different modes.

        - "disabled": synchronized checkpointing will be used.
        - "async": torch.distributed.checkpoint.async_save will be used.
        - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
          separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
          The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
          may degrade due to memory paging. For most users, "async" should suffice as the performance
          overhead is typically small (on the order of tens of seconds) compared to checkpointing
          frequency. This mode can be employed to pursue near-zero checkpointing times
          (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

        "disabled" is the default mode.
        """

        keep_latest_k: int = 10
        """
        Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
        K cannot be 1 as the last one may be in the process of being saved. As a result,
        the metadata of the last one may not be ready yet. The default value is 10 to avoid
        filling up the disk.
        """

        load_step: int = -1
        """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

        exclude_from_loading: list[str] = field(default_factory=list)
        """
        Exclude specific keys from being loaded from the checkpoint.
        Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
        This will load the model only, excluding the specified keys.
        """

        enable_first_step_checkpoint: bool = False
        """
        Enable the checkpoint save at first step. This will save a checkpoint immediately
        after the first step to ensure checkpointing functions correctly. This is useful
        when running on a new cluster or storage to verify checkpointing without waiting
        for many steps or checkpointing too frequently. The default value is False.
        """

        create_seed_checkpoint: bool = False
        """
        Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
        Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
        Could be implemented as a separate script, but this way shares more code.
        """

        load_only: bool = False
        """
        In certain scenarios, you may only need to load checkpoints for verification or debugging
        purposes, without saving any new checkpoints. For example, you might use seed checkpoints
        to validate model correctness. Enabling this option allows checkpoints to be loaded
        without saving any during the training.
        """

        remote_sync_path: str | None = None
        """
        If set, freshly-saved checkpoints are asynchronously pushed to this S3
        prefix via `aws s3 sync`, e.g.
        "s3://my-bucket/path/outputs/.../checkpoint". Each step is uploaded under
        "{remote_sync_path}/step-{step}/". The upload runs on a background thread
        so it does not block training. Only the latest `remote_keep_latest_k`
        checkpoints are retained remotely; older ones are removed with
        `aws s3 rm --recursive`. Requires the `aws` CLI to be available and
        credentials to be configured. The default value is None (disabled).
        """

        remote_keep_latest_k: int = 0
        """
        Keeps only the latest k checkpoints in the remote S3 bucket, purging
        older ones. If 0, defaults to `keep_latest_k`. Only used when
        `remote_sync_path` is set.
        """

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
    ) -> None:
        self.enable = config.enable
        self.load_only = config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        async_mode = config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        if not self.enable:
            return

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None
        self.pg: dist.ProcessGroup | None = None

        self.folder = os.path.join(base_folder, config.folder)

        # Checkpoint policy related fields.
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_allowed_missing_model_key_prefixes = tuple(
            config.initial_load_allowed_missing_model_key_prefixes
        )
        if any(
            not prefix
            for prefix in self.initial_load_allowed_missing_model_key_prefixes
        ):
            raise ValueError(
                "checkpoint.initial_load_allowed_missing_model_key_prefixes "
                "cannot contain an empty prefix."
            )
        self.resume_allowed_missing_optimizer_state_key_prefixes = tuple(
            config.resume_allowed_missing_optimizer_state_key_prefixes
        )
        if any(
            not prefix.startswith("optimizer.state.")
            for prefix in self.resume_allowed_missing_optimizer_state_key_prefixes
        ):
            raise ValueError(
                "checkpoint.resume_allowed_missing_optimizer_state_key_prefixes "
                "must contain only non-empty prefixes beginning with "
                "'optimizer.state.'."
            )
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_path = config.initial_load_path
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                sd_adapter is not None
            ), "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
        self.sd_adapter = sd_adapter
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]
        self.exclude_from_loading = config.exclude_from_loading
        self.interval = config.interval
        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint

        # Async checkpoint related fields.
        async_mode = config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.keep_latest_k = config.keep_latest_k
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

        # Remote (S3) checkpoint sync related fields.
        self.remote_sync_path = config.remote_sync_path
        self.remote_keep_latest_k = (
            config.remote_keep_latest_k
            if config.remote_keep_latest_k > 0
            else self.keep_latest_k
        )
        self.upload_thread: threading.Thread | None = None
        # Tracks the (local_path, step) staged for upload but not yet durable on
        # disk (async modes); flushed to the upload queue on the next save once
        # the prior save future has resolved in _async_wait().
        self.pending_remote_upload: tuple[str, int] | None = None
        if self.remote_sync_path is not None:
            # With node-local checkpoint storage (e.g. /dev/shm) under DCP, each
            # node holds only its own ranks' shards, so every node-local leader
            # (LOCAL_RANK == 0) must upload to assemble the full checkpoint in
            # S3. `aws s3 sync` is additive, so concurrent uploads to the same
            # prefix are safe. Remote purging, however, must be done by a single
            # rank (global rank 0) to avoid concurrent `aws s3 rm` races.
            self.is_upload_leader = int(os.environ.get("LOCAL_RANK", 0)) == 0
            self.is_remote_purge_leader = dist.get_rank() == 0
            if self.is_upload_leader:
                self.upload_queue = queue.Queue()
                self.upload_thread = threading.Thread(
                    target=s3_upload_thread,
                    args=(
                        self.upload_queue,
                        self.remote_sync_path,
                        self.remote_keep_latest_k,
                        self.is_remote_purge_leader,
                    ),
                    daemon=True,
                )
                self.upload_thread.start()

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(f"Unknown checkpoint async_mode {config.async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                # pyrefly: ignore [missing-attribute]
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()
            if (
                hasattr(self, "upload_thread")
                and self.upload_thread
                and self.upload_thread.is_alive()
            ):
                # Flush any deferred async upload, then drain the queue: items
                # already enqueued ahead of Terminate are processed before the
                # worker returns, so no upload is lost on shutdown.
                self._flush_pending_remote_upload()
                self.upload_queue.put(Terminate())
                self.upload_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None
        if to_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(checkpoint_id, "sharded"),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # the reason for only enabling consolidation if there is
                # no mapping is because no mapping implies that we save all fqns
                # to one file. This means we only need one rank to consolidate.
                # Otherwise we should use consolidate_safetensors_files_on_every_rank
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
        allow_partial: bool = False,
        allowed_missing_model_key_prefixes: tuple[str, ...] = (),
        allowed_missing_resume_key_prefixes: tuple[str, ...] = (),
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
            allow_partial (bool): If True, allow missing keys in the checkpoint
                (useful for initial checkpoint loading with optimizer states from
                a different framework where param_groups fields may differ).
            allowed_missing_model_key_prefixes (tuple[str, ...]): Model key
                prefixes authorized to be absent when ``allow_partial`` is true.
            allowed_missing_resume_key_prefixes (tuple[str, ...]): Full state
                dict key prefixes authorized to be absent during a legacy
                checkpoint resume.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(
                hf_state_dict,
                storage_reader=hf_storage_reader,
            )

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            load_kwargs = {}
            if allow_partial:
                checkpoint_metadata = dcp.FileSystemReader(
                    checkpoint_id
                ).read_metadata()
                checkpoint_keys = set(checkpoint_metadata.state_dict_metadata)
                model_keys = set(self.states[MODEL].state_dict())
                missing_model_keys = sorted(model_keys - checkpoint_keys)
                unauthorized_missing_model_keys = [
                    key
                    for key in missing_model_keys
                    if not key.startswith(allowed_missing_model_key_prefixes)
                ]
                if unauthorized_missing_model_keys:
                    missing_key_sample = ", ".join(unauthorized_missing_model_keys[:5])
                    raise ValueError(
                        f"Initial checkpoint {checkpoint_id} is missing "
                        f"{len(unauthorized_missing_model_keys)} unauthorized model "
                        f"keys, including: {missing_key_sample}. Add only intentional "
                        "new-parameter prefixes to "
                        "checkpoint.initial_load_allowed_missing_model_key_prefixes."
                    )

                load_kwargs["planner"] = DefaultLoadPlanner(allow_partial_load=True)
            elif allowed_missing_resume_key_prefixes:
                load_kwargs["planner"] = _SelectiveMissingKeysLoadPlanner(
                    allowed_missing_resume_key_prefixes
                )
            dcp.load(state_dict, checkpoint_id=checkpoint_id, **load_kwargs)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == trainer_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            bool: True if a checkpoint was written (or staged, for async modes) on
            this step.
        """
        if not self._should_save(curr_step, last_step):
            return False

        sl.add_step_tag("checkpoint_save")
        begin = time.monotonic()
        logger.info("Saving the checkpoint (or staging if async is enabled).")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if last_step:
            self._save_last_step(curr_step)
            return True

        states = self._flattened_model_states_sd()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if self.stager is None:
                self.stager = DefaultStager(StagingOptions(True, True, True, True))
            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            assert isinstance(result, AsyncSaveResponse)
            self.save_future = result.upload_completion
            self.staging_future = result.staging_completion
            self.staging = True
            # Files aren't durable until save_future resolves (in the next
            # _async_wait); defer the remote upload until then.
            self._stage_remote_upload(checkpoint_id, curr_step)
        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.save_future = self.dcp_save(
                states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self._stage_remote_upload(checkpoint_id, curr_step)
        else:
            self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )
            # Synchronous save: files are durable now, upload immediately.
            self._enqueue_remote_upload(checkpoint_id, curr_step)
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled) "
            f"in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    @sl.log_trace_span("checkpoint_load")
    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """
        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False

        # Decide between resuming from self.folder and an initial load
        # (initial_load_path / hf_assets). We resume only when self.folder
        # exists AND contains a loadable checkpoint. A folder that exists but
        # holds NO valid checkpoint -- e.g. a run that died before its first
        # save, or whose only save is partial/corrupt (a step-N dir with no
        # finalized .metadata, so _find_load_step() returns -1) -- must NOT
        # silently fall through to the model's random init. Fall back to the
        # initial load path so the model is at least seeded from the intended
        # checkpoint. This matters especially under auto-relaunch, where a job
        # repeatedly killed before its first checkpoint would otherwise
        # cold-start from random weights every time.
        folder_exists = os.path.exists(self.folder)
        if folder_exists:
            resume_step = self._find_load_step() if step == -1 else step
        else:
            resume_step = -1
        do_initial_load = (not folder_exists) or resume_step == -1

        if do_initial_load:
            if folder_exists:
                logger.warning(
                    f"checkpoint.folder {self.folder} exists but contains no loadable "
                    "checkpoint; falling back to the initial load path / hf_assets so "
                    "training does not silently start from random weights."
                )
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized
            if from_hf:
                assert (
                    model_only
                ), "Only model can be loaded when loading from HF's safetensors checkpoint."

            if from_quantized:
                assert (
                    from_hf
                ), "Quantized checkpoint can only be loaded from HuggingFace format."

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                if from_hf:
                    logger.info(
                        f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                    )
            elif from_hf:
                assert (
                    self.sd_adapter is not None
                    and self.sd_adapter.hf_assets_path is not None
                ), "from_hf is True but sd_adapter or hf_assets_path is not provided."
                hf_assets_path = self.sd_adapter.hf_assets_path
                checkpoint_id = hf_assets_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights but the path is not valid. \
                        Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"loading HF safetensors from --model.hf_assets_path: {hf_assets_path}"
                )
            else:
                return False
        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                    f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
                )
            if self.initial_load_in_hf:
                logger.warning(
                    "checkpoint.initial_load_in_hf is True but the checkpoint.folder exists. "
                    "Checkpointer will not load from HF safetensors"
                )
            model_only = resume_step == 0
            checkpoint_id = self._create_checkpoint_id(resume_step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={resume_step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        # Initial loads may allow optimizer-state differences or explicitly
        # authorized new model parameters. Resumes are strict except for a
        # narrowly configured legacy optimizer-state migration.
        is_initial_load = bool(
            do_initial_load
            and self.initial_load_path
            and checkpoint_id == self.initial_load_path
        )
        states = self._states_to_load(model_only, is_initial_load=is_initial_load)
        allow_partial = bool(
            is_initial_load
            and (not model_only or self.initial_load_allowed_missing_model_key_prefixes)
        )
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
            allow_partial=allow_partial,
            allowed_missing_model_key_prefixes=(
                self.initial_load_allowed_missing_model_key_prefixes
                if is_initial_load
                else ()
            ),
            allowed_missing_resume_key_prefixes=(
                self.resume_allowed_missing_optimizer_state_key_prefixes
                if not is_initial_load
                else ()
            ),
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            assert self.staging_future is not None
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            safetensors_metadata_probe = os.path.join(
                folder, filename, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(
        self, model_only: bool, is_initial_load: bool = False
    ) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.
            is_initial_load (bool): Whether this is loading from initial_load_path.
                When True, exclude_from_loading is applied. When False (resume),
                all states are loaded regardless of exclude_from_loading.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        if is_initial_load and self.exclude_from_loading:
            for exclude_key in self.exclude_from_loading:
                if exclude_key not in self.states:
                    raise ValueError(f"{exclude_key} not found in state_dict.")

            states_to_load = {
                k: v
                for k, v in self.states.items()
                if k not in self.exclude_from_loading
            }
        else:
            states_to_load = dict(self.states)

        states_to_load = self._flattened_model_states_sd(states_to_load)

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        checkpoint_id = self._create_checkpoint_id(curr_step)
        self.dcp_save(
            states,
            checkpoint_id=checkpoint_id,
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )
        # Synchronous save: files are durable now, upload immediately.
        self._enqueue_remote_upload(checkpoint_id, curr_step)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
                # The prior async save is now durable on disk; its files can be
                # uploaded safely.
                self._flush_pending_remote_upload()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
                self._flush_pending_remote_upload()
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled."
            )

    def _stage_remote_upload(self, checkpoint_id: str, step: int) -> None:
        """Record an async-saved checkpoint to upload once it is durable.

        For async save modes the on-disk files are not complete until the save
        future resolves (in the next ``_async_wait``), so we defer enqueuing the
        upload until then.
        """
        if self.remote_sync_path is None or not self.is_upload_leader:
            return
        self.pending_remote_upload = (checkpoint_id, step)

    def _flush_pending_remote_upload(self) -> None:
        """Enqueue any deferred async checkpoint upload now that it is durable."""
        if self.pending_remote_upload is not None:
            checkpoint_id, step = self.pending_remote_upload
            self.pending_remote_upload = None
            self._enqueue_remote_upload(checkpoint_id, step)

    def _enqueue_remote_upload(self, checkpoint_id: str, step: int) -> None:
        """Push a durable local checkpoint onto the S3 upload queue."""
        if self.remote_sync_path is None or not self.is_upload_leader:
            return
        self.upload_queue.put((checkpoint_id, step))

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. FTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in _purge_stale_checkpoints.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        )

    def _purge_stale_checkpoints(self):
        if self._should_purge():
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
