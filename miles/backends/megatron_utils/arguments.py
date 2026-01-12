import logging

from megatron.training.arguments import parse_args, validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # always use zero optimizer
    ##############################
    ###########lora###############
    ##############################
    args.use_distributed_optimizer = True

    # from miles.backends.megatron_utils.lora_utils import is_lora_enabled
    # # this should be enalbe after optimize
    # if is_lora_enabled(args):
    #     # Cannot Use distributed optimizer (ZeRO) in LoRA training.
    #     args.use_distributed_optimizer = False
        
    #     # === NEW: Disable features that cause issues with frozen parameters ===
    #     # Disable gradient accumulation fusion (already have --no-gradient-accumulation-fusion)
    #     args.gradient_accumulation_fusion = False
        
    #     # Disable async tensor model parallel allreduce to avoid main_grad access
    #     args.async_tensor_model_parallel_allreduce = False
        
    #     # Disable overlap grad reduce (needs gradient buffers for all params)
    #     args.overlap_grad_reduce = False
        
    #     # Disable sequence parallel if enabled (can cause similar issues)
    #     if hasattr(args, 'sequence_parallel') and args.sequence_parallel:
    #         import logging
    #         logging.getLogger(__name__).warning(
    #             "Disabling sequence_parallel for LoRA training (incompatible with frozen parameters)"
    #         )
    #         args.sequence_parallel = False
    # else:
    #     args.use_distributed_optimizer = True
    ##############################
    ##############################
    ##############################

    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # megatron(dev) optimizer-cpu-offload save ckpt bugs
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
