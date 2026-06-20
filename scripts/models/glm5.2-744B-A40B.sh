MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
N_DENSE_LAYERS=3
N_MOE_LAYERS=75
NHEADS=64

# GLM-5.2 744B-A40B. Same glm_moe_dsa architecture as glm5-744B-A40B.sh (GLM-5.1) and the
# SAME MODEL_ARGS except --rotary-base (8e6 vs 1e6). GLM-5.2 adds DSA *cross-layer index
# sharing* (only "computing" layers 1,2,3,7,11,...,75 carry indexer weights and compute the
# sparse top-k; the rest reuse the most recent computing layer's indices). That schedule
# (index_topk_freq=4 / index_skip_topk_offset=3) is NOT a CLI arg -- it is read from the HF
# config by the model provider (the slime get_glm5_spec for full-FT, or the Megatron-Bridge
# GLM5 provider for LoRA), so it does not appear here.
MODEL_ARGS=(
   --spec "miles_plugins.models.glm5.glm5" "get_glm5_spec"
    --moe-layer-freq "[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}"
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --make-vocab-size-divisible-by 16
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 2048
    --kv-lora-rank 512
    --qk-head-dim 192
    --v-head-dim 256
    --kv-channels 192
    --qk-pos-emb-head-dim 64
    --vocab-size 154880
    --rotary-base 8000000
    --enable-experimental
)
