#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LM-eval harness evaluation for TorchTitan models.

Loads a TorchTitan checkpoint into the model built from a trainer config (or a
model flavor) and runs lm-eval tasks against it through ``TorchTitanEvalLM``
(see lm_eval_wrapper.py): loglikelihood tasks are scored with one packed
block-diagonal prefill and generation tasks use a packed KV-cached decoder
(generate_kv.py). No vLLM or HF conversion is involved.

Usage (the ten paper tasks, one GPU, DCP checkpoint):
    torchrun --nproc_per_node=1 scripts/eval/eval_harness.py \\
        --module olmo2_moe --config olmo2_moe_1b_e8g1t1_8l_midfine \\
        --checkpoint /path/to/step_N \\
        --tokenizer_path ./assets/hf/OLMo-2-0425-1B \\
        --tasks_config scripts/eval/configs/olmo_evals.yaml \\
        --dataset_root /path/to/eval_data \\
        --confirm_run_unsafe_code --batch_size 16 \\
        --output_path results/e8g1t1.json

Usage (a few named tasks, HF checkpoint):
    torchrun --nproc_per_node=1 scripts/eval/eval_harness.py \\
        --module olmo2 --flavor 1B \\
        --checkpoint /path/to/hf_model --hf_checkpoint \\
        --tokenizer_path /path/to/hf_model \\
        --tasks hellaswag,arc_easy
"""

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Ensure the repo root and this directory are importable
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dcp_pickle_compat  # noqa: E402,F401 - reads 3.13-pickled DCP metadata

logger = logging.getLogger(__name__)


def _load_tasks_config(path: str, dataset_root: str | None = None) -> list:
    """Load an lm-eval task spec list from a YAML or JSON file.

    The file's top-level ``tasks`` is returned verbatim and handed to
    ``simple_evaluate(tasks=...)``. lm-eval accepts a ``list[str | dict]``,
    so entries may be plain task names or per-task config dicts carrying
    overrides (``num_fewshot``, ``doc_to_text``, ``generation_kwargs``,
    ``dataset_path``, ...). When ``dataset_root`` is provided, each configured
    ``dataset_path`` is rebased by basename under that root. This lets the same
    task definition run against an equivalent offline dataset staging area.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--tasks_config file not found: {path}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        spec = yaml.safe_load(text)
    elif p.suffix == ".json":
        spec = json.loads(text)
    else:
        raise ValueError(f"--tasks_config must be .yaml/.yml/.json, got '{p.suffix}'")

    if isinstance(spec, list):
        tasks = spec
    elif isinstance(spec, dict) and "tasks" in spec:
        tasks = spec["tasks"]
    else:
        raise ValueError(
            "--tasks_config must be a list of task specs, or a mapping with a "
            "top-level 'tasks' key."
        )
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("--tasks_config 'tasks' must be a non-empty list.")

    if dataset_root is not None:
        root = Path(dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(f"--dataset_root directory not found: {root}")
        rebased_tasks = []
        for task in tasks:
            if not isinstance(task, dict) or "dataset_path" not in task:
                rebased_tasks.append(task)
                continue
            source_path = task["dataset_path"]
            if not isinstance(source_path, str):
                raise ValueError(
                    "task dataset_path must be a string when --dataset_root is set"
                )
            rebased_path = root / Path(source_path).name
            if not rebased_path.is_dir():
                raise FileNotFoundError(
                    f"rebased dataset directory not found: {rebased_path}"
                )
            rebased_tasks.append({**task, "dataset_path": str(rebased_path)})
        tasks = rebased_tasks
    return tasks


def _init_distributed():
    """Initialize distributed (DCP requires it even for single-GPU)."""
    if dist.is_initialized():
        return
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def _pad_params_to_checkpoint(model, checkpoint_path, device):
    """Grow model params whose DCP-saved tensor has extra trailing dim-0 rows.

    Reads the checkpoint's tensor metadata and, for any model parameter whose
    saved size is larger only along dim 0 (all other dims identical), replaces
    the built parameter with a zero-padded one of the saved shape. This handles
    checkpoints saved from a sharded run that padded a table up to a multiple of
    the shard degree (the extra rows are dead and never indexed). A size
    difference in any non-leading dim, or a saved size smaller than the built
    one, is left untouched so DCP still raises on a real mismatch.
    """
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(checkpoint_path).read_metadata()
    saved_sizes = {}
    for key, item in metadata.state_dict_metadata.items():
        size = getattr(item, "size", None)
        if size is not None:
            saved_sizes[key] = tuple(size)

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        saved = saved_sizes.get(name)
        if saved is None:
            continue
        cur = tuple(param.shape)
        if cur == saved:
            continue
        # Only pad when exactly the leading dim grew and the rest is identical.
        if len(cur) == len(saved) and saved[1:] == cur[1:] and saved[0] > cur[0]:
            padded = torch.zeros(saved, dtype=param.dtype, device=device)
            padded[: cur[0]].copy_(param)
            module_path, _, attr = name.rpartition(".")
            submodule = model.get_submodule(module_path) if module_path else model
            submodule.register_parameter(attr, torch.nn.Parameter(padded))
            logger.info(
                f"Padded '{name}' from {cur} to {saved} to match checkpoint "
                "(trailing dead rows from a sharded save)."
            )


def _load_model(args, device, dtype):
    """Build model from ModelSpec and load checkpoint weights.

    Follows the same pattern as the Trainer: build with set_default_dtype so
    parameters are created in the target dtype and no post-hoc .to(dtype) is
    needed (which would corrupt the complex64 RoPE buffer).
    """
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.tools.utils import set_default_dtype

    # Resolve the module's config_registry, which exposes both `model_registry`
    # (flavor -> ModelSpec) and the trainer-config factories (e.g.
    # `stem_dag_1b_midfine`). Core models live under torchtitan.models.{module};
    # experiments (e.g. stem_dag) under torchtitan.experiments.{module}. This is
    # the same resolution ConfigManager uses for --module.
    registry = None
    for prefix in ("torchtitan.models", "torchtitan.experiments"):
        try:
            registry = importlib.import_module(
                f"{prefix}.{args.module}.config_registry"
            )
            break
        except ModuleNotFoundError:
            continue
    if registry is None:
        raise ModuleNotFoundError(
            f"Cannot import config_registry for module '{args.module}' from "
            "torchtitan.models or torchtitan.experiments."
        )

    if args.config is not None:
        # Build the model spec from the exact trainer-config factory used for
        # training (same name as train.py's --config). This guarantees the RoPE
        # config and the attention backend match training. RoPE frequencies are
        # recomputed from config at build time and are NOT stored in the
        # checkpoint (the cache is a non-persistent buffer).
        config_fn = getattr(registry, args.config, None)
        if config_fn is None or not callable(config_fn):
            raise ValueError(
                f"Config function '{args.config}' not found in " f"{registry.__name__}."
            )
        model_spec = config_fn().model_spec
    else:
        model_spec = registry.model_registry(args.flavor)

    model_config = model_spec.model

    # Build on meta device with the target dtype already set, matching Trainer.
    # This ensures parameters are created in the right dtype and no .to(dtype)
    # cast is needed later (avoiding corruption of the complex64 RoPE buffer).
    with torch.device("meta"), set_default_dtype(dtype):
        model = model_config.build()

    model.to_empty(device=device)
    model.init_weights(buffer_device=device)

    # DCP checkpoints may store parameters with extra trailing rows relative to
    # what this (single-process, unsharded) build produces. The Engram
    # vocab-parallel tables, for instance, are padded with dead rows up to a
    # multiple of the training FSDP shard degree (see
    # experiments/engram_olmo2/parallelize.py); the padding rows are never
    # addressed by any index, so they carry no information. DCP load requires an
    # exact shape match, so grow any such parameter's dim-0 to the saved size
    # before loading. Only ever grows the leading dim and only when all trailing
    # dims match exactly -- a genuine shape disagreement still raises at load.
    if not args.hf_checkpoint:
        _pad_params_to_checkpoint(model, args.checkpoint, device)

    checkpoint_config = CheckpointManager.Config(
        enable=True,
        folder="__eval_ckpt__",
        initial_load_path=args.checkpoint,
        initial_load_model_only=True,
        initial_load_in_hf=args.hf_checkpoint,
    )

    sd_adapter = None
    if model_spec.state_dict_adapter is not None:
        sd_adapter = model_spec.state_dict_adapter(
            model_config=model_config,
            hf_assets_path=args.checkpoint,
        )

    checkpointer = checkpoint_config.build(
        dataloader=None,
        model_parts=[model],
        optimizers=None,
        lr_schedulers=None,
        states={},
        sd_adapter=sd_adapter,
    )
    loaded = checkpointer.load()
    if not loaded:
        raise RuntimeError(
            f"Failed to load checkpoint from '{args.checkpoint}'. "
            "Verify the path exists and contains a valid DCP or HF checkpoint."
        )

    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="LM-eval harness evaluation for TorchTitan models"
    )

    # Model selection
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="Model module name (e.g., stem_dag, engram_olmo2, olmo2_moe, olmo2)",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default=None,
        help="Model flavor/size (e.g., 1B, 1B-E8G1T1-8L, debugmodel). Builds the "
        "base ModelSpec via model_registry(flavor). Ignored when --config is "
        "given. Required if --config is omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Trainer-config factory name (same as train.py's --config, e.g. "
        "stem_dag_1b_midfine). Builds the model from that config's "
        "model_spec so RoPE and the attention backend match training "
        "exactly. Overrides --flavor.",
    )

    # Checkpoint and tokenizer
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DCP checkpoint directory or HF model directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to HF tokenizer directory",
    )
    parser.add_argument(
        "--hf_checkpoint",
        action="store_true",
        help="Load checkpoint from HuggingFace safetensors format",
    )

    # Eval tasks
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag",
        help="Comma-separated list of lm-eval task names. For tasks needing "
        "per-task overrides (custom num_fewshot, doc_to_text, "
        "generation_kwargs, e.g. 'drop'), use --tasks_config instead.",
    )
    parser.add_argument(
        "--tasks_config",
        type=str,
        default=None,
        help="Path to a YAML/JSON file whose top-level 'tasks' is an lm-eval "
        "task spec list (names and/or per-task config dicts). Passed "
        "verbatim to simple_evaluate(tasks=...). Overrides --tasks. See "
        "scripts/eval/configs/olmo_evals.yaml.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional offline dataset root. Rebase every dataset_path in "
        "--tasks_config by basename under this directory.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples per task (int for count, float<1 for fraction)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the padded loglikelihood fallback",
    )

    # Model config
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length (per-request truncation length)",
    )
    parser.add_argument(
        "--gen_max_tokens",
        type=int,
        default=65536,
        help="Token budget for packed batching (GQAttention models). Prompts "
        "are greedily packed until their summed (len + max_gen_len) would "
        "exceed this. Lower it (e.g. 8192) to reduce peak memory / avoid "
        "OOM; raise it for better throughput. Does not affect the padded "
        "loglikelihood fallback, which uses --batch_size.",
    )

    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual samples in results",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show tqdm progress bars during evaluation",
    )

    # Code-eval tasks (mbpp, humaneval): these generate and then EXECUTE model
    # output, so lm-eval gates them behind an explicit opt-in.
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Allow tasks marked UNSAFE_CODE (e.g. mbpp, humaneval) that execute "
        "generated code. Also sets HF_ALLOW_CODE_EVAL=1 for the code_eval "
        "metric. Only enable on trusted/sandboxed hosts.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help="Comma-separated generation kwargs forwarded to generate_until "
        "tasks, e.g. 'max_gen_toks=512,until=\\n\\n'. Parsed by lm-eval.",
    )

    args = parser.parse_args()
    if (args.config is None) == (args.flavor is None):
        parser.error(
            "exactly one of --config or --flavor is required: --config "
            "<factory> to match a training config, or --flavor <size> for "
            "the base model spec."
        )
    if args.dataset_root and not args.tasks_config:
        parser.error("--dataset_root requires --tasks_config")
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    model_selector = f"config={args.config}" if args.config else f"flavor={args.flavor}"
    logger.info(f"Loading model (module={args.module}, {model_selector})")
    model = _load_model(args, device, dtype)
    logger.info("Model loaded")

    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer(tokenizer_path=args.tokenizer_path)

    from lm_eval_wrapper import TorchTitanEvalLM

    eval_lm = TorchTitanEvalLM(
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gen_max_tokens=args.gen_max_tokens,
        show_progress=args.show_progress,
    )

    if args.tasks_config:
        tasks = _load_tasks_config(args.tasks_config, args.dataset_root)
        task_names = [
            t if isinstance(t, str) else t.get("task", "<dict>") for t in tasks
        ]
        logger.info(
            f"Running evaluation on tasks from {args.tasks_config}: {task_names}"
        )
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
        logger.info(f"Running evaluation on tasks: {tasks}")

    if args.confirm_run_unsafe_code:
        # Execution-based code-eval tasks (mbpp, humaneval) use lm-eval's
        # internal code_eval metric, which refuses to run generated programs
        # unless HF_ALLOW_CODE_EVAL=1. Only enable on trusted/sandboxed hosts.
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model=eval_lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info("=== Results ===")
        for task_name, task_results in results.get("results", {}).items():
            logger.info(f"  {task_name}: {task_results}")

        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            safe_keys = [
                "results",
                "versions",
                "n-shot",
                "higher_is_better",
                "n-samples",
                "config",
            ]
            serializable = {k: v for k, v in results.items() if k in safe_keys}
            with open(args.output_path, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"Results saved to {args.output_path}")

            # When --log_samples is set, persist the per-doc samples too: one
            # JSON file per task next to the results file. lm-eval returns them
            # under results["samples"] (keyed by task name); each example carries
            # the full `doc` and `filtered_resps` (the generation). Docs may hold
            # numpy/tensor values, so encode with lm-eval's
            # handle_non_serializable fallback.
            if args.log_samples and "samples" in results:
                from lm_eval.utils import handle_non_serializable

                samples_dir = os.path.splitext(args.output_path)[0] + "_samples"
                os.makedirs(samples_dir, exist_ok=True)
                for task_name, task_samples in results["samples"].items():
                    sample_path = os.path.join(samples_dir, f"{task_name}.json")
                    with open(sample_path, "w") as f:
                        json.dump(
                            task_samples,
                            f,
                            indent=2,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                logger.info(f"Samples saved to {samples_dir}/")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
