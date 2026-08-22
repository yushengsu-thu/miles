"""Serve full GLM-5.2 Multi-LoRA through the official Tinker SDK frontend.

This launcher pins the first full-model API smoke to five whole 8xH200
nodes: four BF16 trainer nodes and one disaggregated FP8 SGLang node.  The
trainer layout is TP8 / EP32 / DP4 / PP1 / CP1 / ETP1.

The smoke intentionally omits rollout routing replay.  It validates the
``tinker==0.24.1`` SFT/operation/publish/sample contract, not on-policy GLM
RL equivalence.  The public JSON frontend cannot currently carry GLM routed
expert replay metadata.

Args:
    --hf-checkpoint: Native 78-layer BF16 ``zai-org/GLM-5.2`` checkpoint.
    --sglang-config: One-engine, 8-GPU FP8 rollout deployment YAML.
    --megatron-path: Megatron-LM checkout used by the Ray runtime.
    --save-dir: Per-rank operation checkpoint and adapter sidecar directory.

The Ray cluster must already span all five nodes::

    MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<ray-head-ip> python \\
      examples/multi_lora_operations/run_glm52_full_multi_lora_operations.py \\
      serve-tinker --megatron-path /personal/glm52-tinker/Megatron-LM

Run the dual-client unmodified SDK smoke on the Ray head after
``/api/v1/healthz`` is ready::

    python tests/e2e/tinker_frontend/glm52_full_dual_client_sdk_smoke.py \\
      --base-url http://127.0.0.1:8068 \\
      --base-model /cluster-storage/models/GLM-5.2 \\
      --out-dir /scratch/glm52-tinker-sdk-smoke
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_SGLANG_CONFIG = Path(U.repo_base_dir) / "examples" / "multi_lora_operations" / "glm52_full_sglang_fp8.yaml"
_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
_FULL_NUM_LAYERS = 78
_FULL_NUM_EXPERTS = 256
_FULL_EXPERTS_PER_TOKEN = 8


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    """Checkpoint, topology, and service knobs for the five-node smoke."""

    run_id: str = U.create_run_id()

    hf_checkpoint: str = "/cluster-storage/models/GLM-5.2"
    sglang_config: str = str(_DEFAULT_SGLANG_CONFIG)
    save_dir: str = "/scratch/glm52_full_multi_lora_tinker"
    megatron_path: str = "/root/Megatron-LM"

    actor_num_nodes: int = 4
    num_gpus_per_node: int = 8
    rollout_num_gpus: int = 8
    rollout_num_gpus_per_engine: int = 8

    # Storage/deployment ceiling. SDK clients use logical rank 8 or 16.
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: str = _TARGET_MODULES
    n_adapters: int = 2

    backend_batch_size: int = 16
    sequence_length: int = 8192
    max_tokens_per_gpu: int = 8192
    sglang_mem_fraction_static: float = 0.85
    sglang_lora_backend: str = "triton"
    sglang_router_port: int = 30080

    api_port: int = 8068
    enable_wandb: bool = False


def _read_full_checkpoint_config(path: str) -> dict:
    checkpoint = Path(path)
    if "5layer" in checkpoint.name.lower() or "5-layer" in checkpoint.name.lower():
        raise ValueError(f"reduced GLM-5.2 checkpoints are forbidden for this smoke: {checkpoint}")
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"full GLM-5.2 config not found: {config_path}")
    with config_path.open() as config_file:
        return json.load(config_file)


def _validate_full_checkpoint(path: str, *, require_fp8: bool = False) -> None:
    config = _read_full_checkpoint_config(path)
    if "auto_map" in config:
        raise ValueError(f"{Path(path) / 'config.json'} must not contain auto_map; use the native GLM-5.2 checkpoint")
    actual = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "n_routed_experts": config.get("n_routed_experts"),
    }
    expected_architecture = "GlmMoeDsaForCausalLM"
    if (
        actual["model_type"] != "glm_moe_dsa"
        or expected_architecture not in (actual["architectures"] or [])
        or actual["num_hidden_layers"] != _FULL_NUM_LAYERS
        or actual["num_experts_per_tok"] != _FULL_EXPERTS_PER_TOKEN
        or actual["n_routed_experts"] != _FULL_NUM_EXPERTS
    ):
        raise ValueError(
            f"expected native full zai-org/GLM-5.2 (glm_moe_dsa, {expected_architecture}, {_FULL_NUM_LAYERS} layers, top-{_FULL_EXPERTS_PER_TOKEN} of {_FULL_NUM_EXPERTS} routed experts), got {actual} from {Path(path) / 'config.json'}"
        )
    quantization = config.get("quantization_config")
    if require_fp8:
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
            raise ValueError(
                f"the rollout checkpoint must declare quantization_config.quant_method='fp8': {Path(path) / 'config.json'}"
            )
    elif quantization is not None:
        raise ValueError(
            f"the trainer checkpoint must be unquantized (Miles runs it in BF16), got quantization_config={quantization}"
        )


def _rollout_checkpoint(args: ScriptArgs) -> str:
    config_path = Path(args.sglang_config)
    if not config_path.is_file():
        raise FileNotFoundError(f"SGLang rollout config not found: {config_path}")
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    models = config.get("sglang") if isinstance(config, dict) else None
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError(f"{config_path} must define exactly one SGLang model")
    model = models[0]
    groups = model.get("server_groups")
    if model.get("name") != "default" or model.get("update_weights") is not True:
        raise ValueError(f"{config_path} must define an updateable model named 'default'")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError(f"{config_path} must define exactly one rollout server group")
    group = groups[0]
    effective_gpus_per_engine = group.get("num_gpus_per_engine", model.get("num_gpus_per_engine"))
    if (
        group.get("worker_type") != "regular"
        or group.get("num_gpus") != args.rollout_num_gpus
        or effective_gpus_per_engine != args.rollout_num_gpus_per_engine
    ):
        raise ValueError(f"{config_path} must define one regular {args.rollout_num_gpus}-GPU rollout engine")
    model_path = model.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError(f"{config_path} must set the FP8 rollout model_path")
    return model_path


def _validate_topology(args: ScriptArgs) -> None:
    if args.actor_num_nodes != 4 or args.num_gpus_per_node != 8:
        raise ValueError(
            f"the first full Tinker smoke is pinned to four 8xH200 trainer nodes; got {args.actor_num_nodes}x{args.num_gpus_per_node}"
        )
    if args.rollout_num_gpus != 8 or args.rollout_num_gpus_per_engine != 8:
        raise ValueError("the FP8 rollout plane is pinned to one 8xH200 engine")
    if args.n_adapters < 2:
        raise ValueError("the dual-client smoke requires at least two adapter slots")
    if args.lora_rank != 16:
        raise ValueError(f"the dual-client smoke pins the storage and serving rank to 16; got {args.lora_rank}")

    actor_world_size = args.actor_num_nodes * args.num_gpus_per_node
    if actor_world_size != 32 or _FULL_NUM_EXPERTS % actor_world_size:
        raise ValueError(
            f"GLM-5.2's {_FULL_NUM_EXPERTS} routed experts require EP32 for this smoke; got EP={actor_world_size}"
        )


def _sglang_args(args: ScriptArgs) -> str:
    return (
        "--pause-generation-mode in_place "
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-enable-dp-attention "
        f"--sglang-ep-size {args.rollout_num_gpus_per_engine} "
        f"--sglang-dp-size {args.rollout_num_gpus_per_engine} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-page-size 64 "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        f"--sglang-context-length {args.sequence_length} "
        "--sglang-cuda-graph-max-bs 64 "
        "--sglang-max-running-requests 128 "
        "--sglang-chunked-prefill-size 16384 "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-shared-experts-fusion "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
        f"--sglang-router-port {args.sglang_router_port} "
        f"--sglang-config {args.sglang_config} "
    )


def _train_args(args: ScriptArgs) -> str:
    actor_world_size = args.actor_num_nodes * args.num_gpus_per_node
    checkpoint_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge --dsa-attention-backend tilelang "
    )
    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 --target-modules "{args.target_modules}" --no-gradient-accumulation-fusion '
    operation_args = f"--tinker-backend --tinker-frontend --multi-lora-n-adapters {args.n_adapters} --multi-lora-api-port {args.api_port} --multi-lora-idle-poll-s 1 --tinker-max-coalesce-wait-s 30 --tinker-max-empty-wait-s 1 --tinker-sampling-max-context {args.sequence_length} "
    batch_args = f"--rollout-batch-size {args.backend_batch_size} --n-samples-per-prompt 1 --seq-length {args.sequence_length} --rollout-max-context-len {args.sequence_length} --global-batch-size {args.actor_num_nodes} --num-rollout 1000000 --use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    optimizer_args = (
        "--optimizer adam --lr 1e-4 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 "
    )
    parallel_args = (
        f"--tensor-model-parallel-size {args.num_gpus_per_node} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {actor_world_size} "
        "--expert-tensor-parallel-size 1 "
        "--qkv-format thd "
        "--micro-batch-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
    )
    topology_args = f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} --rollout-num-gpus {args.rollout_num_gpus} "
    save_args = f"--save {args.save_dir} --save-interval 1000000 "
    numeric_args = "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""
    return "".join(
        (
            checkpoint_args,
            lora_args,
            operation_args,
            batch_args,
            optimizer_args,
            parallel_args,
            topology_args,
            _sglang_args(args),
            save_args,
            numeric_args,
            wandb_args,
            " ",
        )
    )


def _runtime_env() -> dict[str, str]:
    runtime_env = {
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
        "MILES_USE_LEGACY_ROLLOUT_V1": "0",
        "MILES_EXPERIMENTAL_FT_TRAINER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
    }
    if api_key := os.environ.get("MILES_TINKER_API_KEY"):
        runtime_env["MILES_TINKER_API_KEY"] = api_key
    return runtime_env


def _live_ray_nodes() -> list[dict]:
    import ray

    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    try:
        return ray.nodes()
    finally:
        ray.shutdown()


def _validate_external_ray_nodes(nodes: list[dict]) -> None:
    gpu_nodes = []
    h200_nodes = []
    for node in nodes:
        resources = node.get("Resources", {})
        if not node.get("Alive") or resources.get("GPU", 0) <= 0:
            continue
        gpu_nodes.append(node)
        is_h200 = any(key.startswith("accelerator_type:") and "H200" in key.upper() for key in resources)
        if resources.get("GPU") == 8 and is_h200:
            h200_nodes.append(node)
    if len(gpu_nodes) != len(h200_nodes):
        raise RuntimeError("every live GPU node in the external Ray cluster must be one whole 8xH200 node")
    if len(h200_nodes) < 5:
        raise RuntimeError(
            f"this launcher requires at least five live whole 8xH200 Ray nodes (40 H200 GPUs); found {len(h200_nodes)}"
        )


def _require_external_ray() -> None:
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise RuntimeError(
            "this five-node launcher requires an existing 40-GPU Ray cluster; set MILES_SCRIPT_EXTERNAL_RAY=1 before starting it"
        )
    _validate_external_ray_nodes(_live_ray_nodes())


@app.command("serve-tinker")
@U.dataclass_cli
def serve_tinker(args: ScriptArgs) -> None:
    """Start four BF16 trainer nodes plus one FP8 rollout node."""

    _require_external_ray()
    _validate_topology(args)
    _validate_full_checkpoint(args.hf_checkpoint)
    rollout_checkpoint = _rollout_checkpoint(args)
    _validate_full_checkpoint(rollout_checkpoint, require_fp8=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[run] full GLM-5.2 Multi-LoRA Tinker service: 4x8 H200 trainer + 1x8 H200 rollout, TP8/EP32/DP4/PP1/CP1/ETP1, slots={args.n_adapters}, storage-rank={args.lora_rank}, frontend=http://127.0.0.1:{args.api_port}",
        flush=True,
    )
    U.execute_train(
        train_args=_train_args(args),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="glm5.2-744B-A40B_lora",
        train_script="train_multi_lora_operations.py",
        megatron_path=args.megatron_path,
        extra_env_vars=_runtime_env(),
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
