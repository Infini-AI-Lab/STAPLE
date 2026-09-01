# Evaluation

`scripts/eval/eval_harness.py` scores a TorchTitan checkpoint with
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) without
converting it to HuggingFace format. It builds the model from the same trainer
config used for training, loads the DCP checkpoint, and plugs it into lm-eval
through `TorchTitanEvalLM` (`scripts/eval/lm_eval_wrapper.py`).

## Install

The eval dependencies are not needed for training and live in their own file:

```bash
pip install -r scripts/eval/requirements.txt
```

## The ten tasks

`scripts/eval/configs/olmo_evals.yaml` defines the suite reported in the paper:

| Task | Type | Notes |
|---|---|---|
| hellaswag, winogrande, arc_easy, arc_challenge, mmlu | loglikelihood | lm-eval defaults |
| gsm8k, hendrycks_math | generation | lm-eval defaults |
| mbpp, humaneval | generation + code execution | need `--confirm_run_unsafe_code` |
| drop | generation | 3-shot, custom prompt and stop strings (see the YAML) |

## Running

Pick the model with `--module` and the trainer config the checkpoint was trained
with (`--config`, same name as `torchtitan.train --config`). `--tokenizer_path`
is the OLMo-2 HF directory used as `--hf-assets-path` during training.

```bash
EVAL_DATA=/path/to/eval_data
TOK=/path/to/olmo2-1b-base     # HF OLMo-2 tokenizer directory

# STAPLE-12L
torchrun --nproc_per_node=8 scripts/eval/eval_harness.py \
    --module stem_dag --config stem_dag_1b_midfine \
    --checkpoint outputs/checkpoint/step-N \
    --tokenizer_path $TOK \
    --tasks_config scripts/eval/configs/olmo_evals.yaml --dataset_root $EVAL_DATA \
    --confirm_run_unsafe_code --batch_size 16 \
    --output_path results/staple_12l.json
```

Other models only change `--module` / `--config`:

| Model | `--module` | `--config` |
|---|---|---|
| STAPLE-12L / L4 / low-rank r512 / 7B | `stem_dag` | `stem_dag_1b_midfine`, `stem_dag_1b_l4_midfine`, `stem_dag_lowrank_1b_r512_midfine`, `stem_dag_7b_midfine` |
| Engram (STAPLE-12L size) | `engram_olmo2` | `engram_olmo2_1b_stem_dag_l12_midfine` |
| MoE E8G1T1-8L | `olmo2_moe` | `olmo2_moe_1b_e8g1t1_8l_midfine` |
| Dense OLMo-2 baseline | `olmo2` | `olmo2_1b_midfine`, `olmo2_7b_midfine` |
