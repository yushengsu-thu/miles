import json
from argparse import Namespace
from pathlib import Path

LORA_ADAPTER_NAME = "miles_lora"


def is_lora_weight_name(name: str) -> bool:
    """Check if an HF weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def lora_rollout_enabled(args: Namespace) -> bool:
    """LoRA enabled AND the rollout side participates; false under --lora-train-only.

    Gates everything rollout-facing: SGLang's ``enable_lora``, the per-request
    ``lora_path``, and the adapter weight sync. Training-side LoRA is unaffected.
    """
    return is_lora_enabled(args) and not getattr(args, "lora_train_only", False)


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


# Qwen3.5 / 3.6 (hybrid GDN + MoE behind a VL wrapper, plus an MTP block): Megatron-anchored patterns keep the adapters
# off the MTP block (no adapter export mapping) and the vision tower; the pattern leaves map to the HF names SGLang
# serves (in_proj -> in_proj_qkvz + in_proj_ba). Shared by scripts/run_qwen3_5_35b_a3b_lora.py and the Tinker gateway.
QWEN3_5_LAYERS = "language_model.decoder.layers.*"
QWEN3_5_ATTENTION_TARGETS = (
    f"{QWEN3_5_LAYERS}.self_attention.linear_qkv",
    f"{QWEN3_5_LAYERS}.self_attention.linear_proj",
)
QWEN3_5_GDN_TARGETS = (f"{QWEN3_5_LAYERS}.self_attention.in_proj", f"{QWEN3_5_LAYERS}.self_attention.out_proj")
QWEN3_5_MOE_MLP_TARGETS = tuple(
    f"{QWEN3_5_LAYERS}.mlp.{leaf}"
    for leaf in ("experts.linear_fc1", "experts.linear_fc2", "shared_experts.linear_fc1", "shared_experts.linear_fc2")
)
QWEN3_5_DENSE_MLP_TARGETS = (f"{QWEN3_5_LAYERS}.mlp.linear_fc1", f"{QWEN3_5_LAYERS}.mlp.linear_fc2")
QWEN3_5_OUTPUT_TARGET = "language_model.output_layer"


def qwen3_5_lora_target_modules(
    *, moe: bool, train_attn: bool = True, train_mlp: bool = True, train_unembed: bool = False
) -> list[str]:
    """LoRA targets for Qwen3.5/3.6 by module group: attention + GDN projections, routed/shared (or dense) MLP, output layer."""
    modules: list[str] = []
    if train_attn:
        modules.extend(QWEN3_5_ATTENTION_TARGETS)
    if train_mlp:
        modules.extend(QWEN3_5_MOE_MLP_TARGETS if moe else QWEN3_5_DENSE_MLP_TARGETS)
    if train_attn:
        modules.extend(QWEN3_5_GDN_TARGETS)
    if train_unembed:
        modules.append(QWEN3_5_OUTPUT_TARGET)
    return modules


def save_adapter_to_disk(out_dir, config: dict, tensors: dict) -> None:
    """Write a LoRA adapter dir (adapter_config.json + adapter_model.safetensors)."""
    import safetensors.torch  # lazy: this module is imported on paths that never touch weights

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    safetensors.torch.save_file(tensors, str(out / "adapter_model.safetensors"))
