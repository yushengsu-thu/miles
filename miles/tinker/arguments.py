import argparse
import logging

from miles.utils.hf_config import load_hf_config
from miles.utils.lora import qwen3_5_lora_target_modules

logger = logging.getLogger(__name__)


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    add_argument(
        "session-ttl-s",
        type=float,
        default=3600.0,
        help="Idle seconds before a recorded /oai/sessions/{sid} is swept, the safety net for agent trials that die before DELETE (default: 3600)",
    )
    add_argument("train-attn", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-mlp", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-unembed", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert (
        args.target_modules is None and args.exclude_modules is None
    ), "Tinker uses --tinker-train-attn/mlp/unembed; --target-modules and --exclude-modules are not supported"
    hf_config = load_hf_config(args.hf_checkpoint)
    modules = _resolve_target_modules(
        hf_config,
        train_attn=args.tinker_train_attn,
        train_mlp=args.tinker_train_mlp,
        train_unembed=args.tinker_train_unembed,
    )
    # The common LoRA validator parses and validates this before trainer/engine initialization.
    args.target_modules = ",".join(modules)
    _configure_gdn_batching(args, hf_config)


def _configure_gdn_batching(args, hf_config):
    """GatedDeltaNet models (Qwen3.5/3.6, Qwen3-Next): megatron-core rejects packed (thd) sequences, so train one unpacked sequence per micro-batch; multi-LoRA accepts exactly that shape."""
    text_config = getattr(hf_config, "text_config", None) or hf_config
    if "linear_attention" not in (getattr(text_config, "layer_types", None) or ()):
        return
    if args.qkv_format != "bshd" or args.micro_batch_size != 1 or args.use_dynamic_batch_size:
        logger.info(
            "GatedDeltaNet model: using --qkv-format bshd --micro-batch-size 1 without --use-dynamic-batch-size"
        )
    args.qkv_format = "bshd"
    args.micro_batch_size = 1
    args.use_dynamic_batch_size = False


def _resolve_target_modules(hf_config, *, train_attn, train_mlp, train_unembed):
    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        modules = qwen3_5_lora_target_modules(
            moe=hf_config.model_type == "qwen3_5_moe",
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        assert modules, "Tinker requires at least one trainable LoRA module group"
        return modules
    # Other architectures need their own complete attention/MLP mapping.
    assert hf_config.model_type in (
        "qwen3",
        "qwen3_moe",
    ), f"Tinker target layout is not defined for model_type={hf_config.model_type!r}"
    modules = []
    if train_attn:
        modules.extend(("q_proj", "k_proj", "v_proj", "o_proj"))
    if train_mlp:
        modules.extend(("gate_proj", "up_proj", "down_proj"))
    if train_unembed:
        modules.append("lm_head")
    assert modules, "Tinker requires at least one trainable LoRA module group"
    return modules
