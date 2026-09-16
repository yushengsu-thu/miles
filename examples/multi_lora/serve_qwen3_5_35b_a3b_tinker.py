"""Prepare Qwen3.5-35B-A3B and serve the Tinker gateway on separate training and sampling GPUs (hybrid GDN + MoE: bshd static micro-batches, per-expert adapters, LoRA off the MTP block)."""

from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)

    hf_checkpoint: str | None = None
    model_name: str = "Qwen3.5-35B-A3B"  # or Qwen3.6-35B-A3B: same architecture and HF classes
    model_dir: str = "/root/models"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_gpus: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2  # num_query_groups=2 caps tensor parallelism for the 35B-A3B geometry
    ep: int = 4

    # LoRA slot pool; per-client rank comes from the SDK, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 64
    n_adapters: int = 4  # per-expert adapters: at most 1023 // (256 // ep) slots fit one grouped GEMM
    tinker_train_attn: bool = True
    tinker_train_mlp: bool = True
    tinker_train_unembed: bool = True

    tinker_port: int = 10613
    tinker_base_model: str = "Qwen/Qwen3.5-35B-A3B"
    max_tokens_per_gpu: int = 32768  # also the gateway's per-datum cap (one sequence per micro-batch under bshd)
    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.7

    extra_args: str = ""

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = f"{self.output_dir}/checkpoints"
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"

    @property
    def model_type(self) -> str:
        return {"Qwen3.5-35B-A3B": "qwen3.5-35B-A3B_lora", "Qwen3.6-35B-A3B": "qwen3.6-35B-A3B_lora"}[self.model_name]


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint. Run once per node before serving."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Serve the Tinker gateway (idles until clients connect)."""
    print(
        f"[run] tinker gateway ({args.model_name}): {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs, "
        f"{args.n_adapters} adapter slots, port {args.tinker_port}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        "--no-gradient-accumulation-fusion "
        f"--multi-lora-n-adapters {args.n_adapters} "
    )

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} --tinker-base-model {args.tinker_base_model} "
        f"--tinker-checkpoint-root {args.save_dir}/{args.run_id}"
    )

    for group in ("attn", "mlp", "unembed"):
        enabled = getattr(args, f"tinker_train_{group}")
        tinker_args += f" --{'' if enabled else 'no-'}tinker-train-{group}"

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-4 "

    # bshd + micro-batch 1: megatron-core GatedDeltaNet rejects packed (thd) sequences, so no dynamic batching
    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--qkv-format bshd --micro-batch-size 1 --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 --sglang-lora-backend triton "
    )

    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    misc_args = "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {topology_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
