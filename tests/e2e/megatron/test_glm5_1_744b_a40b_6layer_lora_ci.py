import os

from scripts.run_glm5_1_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for scripts/run_glm5_1_744b_a40b_lora.py (LoRA-via-Megatron-Bridge) on the
# 6-layer toy: full rollout -> train -> save loop with a tiny gsm8k rollout, exercising the
# DSA bridge-LoRA path (indexer excluded from LoRA targets). Runs the MoE-expert LoRA
# combination matrix — {shared-outer + virtual-experts} and {per-expert +
# no-virtual-experts}, each on both DSA kernel backends (tilelang / megatron) —
# sequentially; EVERY combination must pass. Verifies the script is functional, not model
# accuracy. Uses 4 of the suite's GPUs (TP=EP=4).


register_cuda_ci(est_time=5400, suite="stage-c-8-gpu-h100", labels=["model-scripts"])

# The weight-update checker (auto-on under --ci-test) must skip the engine-side stacked
# params that a frozen-base LoRA run cannot re-ship: sglang stacks q_a+kv_a into
# fused_qkv_a_proj_with_mqa at load, and the DSA indexer weights likewise have no 1:1
# trainer export. They keep their (correct) checkpoint values; everything else is verified.
_BASE_EXTRA = (
    "--ci-test "
    "--ci-disable-logprobs-checker "
    "--disable-weights-backuper "
    "--check-weight-update-skip-list fused_qkv_a_proj_with_mqa indexer. "
)

# (name, dsa_attention_backend, experts_shared_outer_loras, virtual_experts_serving)
_CONFIGS = [
    ("tilelang + shared-outer + virtual-experts", "tilelang", True, True),
    ("megatron + shared-outer + virtual-experts", "megatron", True, True),
    ("tilelang + per-expert + no-virtual-experts", "tilelang", False, False),
    ("megatron + per-expert + no-virtual-experts", "megatron", False, False),
]


def _args(dsa: str, shared_outer: bool, virtual_experts: bool) -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.1-6layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=1,
        enable_wandb=False,
        dsa_attention_backend=dsa,
        experts_shared_outer_loras=shared_outer,
        extra_args=_BASE_EXTRA + ("" if virtual_experts else "--no-sglang-lora-use-virtual-experts "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(_args(*_CONFIGS[0][1:]))
    for name, dsa, shared_outer, virtual_experts in _CONFIGS:
        print(f"[glm5.1-lora-ci] ===== combo: {name} =====", flush=True)
        # a fresh ray/sglang between combos; the previous combo's teardown can lag
        U.exec_command("ray stop --force || true; pkill -9 sglang || true; sleep 10")
        execute(_args(dsa, shared_outer, virtual_experts))
        print(f"[glm5.1-lora-ci] ===== combo PASSED: {name} =====", flush=True)
