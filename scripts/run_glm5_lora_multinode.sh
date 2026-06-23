#!/usr/bin/env bash
# ============================================================================
#  run_glm5_lora_multinode.sh
#    Multi-node GLM-5.x LoRA RL launcher — a thin wrapper around
#    scripts/run_glm5_lora.py that drives the *multi-node* flow.
#
#    VALIDATED 2026-06-23: GLM-5.2_5layer, 2 nodes x 4xH200, unfused
#    (--dsa-attention-backend megatron-bridge), 50 steps -> Ray job SUCCEEDED.
#    Topology realized: world_size=8, TP4/EP4/PP1 (intra-node) + DP2 (cross-node),
#    qkv-format bshd; live sglang colocate rollout on both nodes; 5 LoRA adapter
#    checkpoints saved (save-interval 10).
#
#  WHY THIS WRAPPER EXISTS
#    run_glm5_lora.py is single-node by design: its _train() hardcodes
#    `--actor-num-nodes 1`, and U.execute_train only does a local
#    `ray start --head` (no worker join). Rather than edit the launcher, this
#    wrapper:
#      (a) forms a Ray cluster across the nodes MANUALLY (head + workers), and
#      (b) submits the job with MILES_SCRIPT_EXTERNAL_RAY=1 so miles reuses the
#          existing cluster, overriding --actor-num-nodes via --extra-args
#          (argparse last-wins: extra_args is appended after misc_args in
#          run_glm5_lora.py:_train, so "--actor-num-nodes $NUM_NODES" wins).
#
#  ⚠ CRITICAL — set GPUS_PER_NODE to the *REAL* per-node GPU count.
#    miles' rollout address allocator (miles/ray/rollout/addr_allocator.py) uses
#    `--num-gpus-per-node` to decide which physical node each sglang engine lives
#    on (num_engines_per_node = num_gpus_per_node // rollout_num_gpus_per_engine).
#    miles' DEFAULT is 8. If the real node has 4 GPUs but this stays 8, miles
#    thinks all engines are on node 0 and hands every engine the HEAD node's
#    dist_init_addr -> worker-node sglang engines block on a cross-node TCPStore
#    rendezvous to the head and DIE with a 600s socket timeout. So this wrapper
#    passes `--num-gpus-per-node $GPUS_PER_NODE` in BOTH the script flag and
#    --extra-args (the latter sets the miles/train.py value the allocator reads).
#
#  PARALLELISM RATIONALE (keep TP intra-node, cross nodes with DP)
#    TP=EP=GPUS_PER_NODE stays inside one node's NVLink domain; PP=1 (so the
#    GLM-5.2 cross-layer DSA build-time PP-split assert is a no-op); the extra
#    node(s) add pure DP (gradient all-reduce), which tolerates inter-node
#    bandwidth. Do NOT let TP span nodes.
#
#  PREREQUISITES
#    * A shared filesystem for code + checkpoints (e.g. /personal) reachable by
#      all nodes, and the model present on every node (e.g. /cluster-storage).
#    * The editable installs done on every node's container (sglang,
#      Megatron-Bridge on the `bridge`/`bridge-dev-glm` branch, miles).
#    * Inter-node TCP connectivity; the inter-node NIC name (NCCL_IFNAME).
#
#  USAGE  (run from a miles checkout; this script cd's to the miles root itself)
#    # 1. On the HEAD node — start the Ray head and wait for workers to join:
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=4 NUM_NODES=2 \
#      bash scripts/run_glm5_lora_multinode.sh head
#
#    # 2. On EACH WORKER node — join the head:
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=4 \
#      bash scripts/run_glm5_lora_multinode.sh worker
#
#    # 3. Back on the HEAD node, once `ray status` shows NUM_NODES nodes — launch.
#    #    wandb is ON by default; export WANDB_API_KEY to actually log (else miles skips it):
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=4 NUM_NODES=2 MODEL_NAME=GLM-5.2_5layer \
#      DSA_BACKEND=megatron-bridge NUM_ROLLOUT=50 \
#      WANDB=on WANDB_API_KEY=xxxxxxxx WANDB_TEAM=my-team \
#      bash scripts/run_glm5_lora_multinode.sh launch
#    #    (WANDB=offline -> local only; WANDB=off -> no wandb. See the wandb knobs below.)
#    #    DAPO-Math instead of gsm8k (run prepare with --task dapo-math first; long CoT):
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=4 NUM_NODES=2 MODEL_NAME=GLM-5.2_5layer \
#      TASK=dapo-math RESP_LEN=4096 \
#      bash scripts/run_glm5_lora_multinode.sh launch   # add DAPO_DYNAMIC_SAMPLING=on on a real model
#
#    # 4. Monitor (head node):
#    RAY_ADDRESS=http://$HEAD_IP:8265 ray job logs   $JOB_ID --follow
#    RAY_ADDRESS=http://$HEAD_IP:8265 ray job status $JOB_ID
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" && "$ROLE" != "launch" ]]; then
  echo "usage: $0 {head|worker|launch}   (see header for the full flow)" >&2
  exit 2
fi

# cd to the miles repo root so `scripts/run_glm5_lora.py` resolves (script lives in scripts/).
cd "$(dirname "$0")/.."

# ----- knobs (env-overridable; defaults = the validated 2-node GLM-5.2_5layer run) -----
HEAD_IP="${HEAD_IP:?set HEAD_IP to the head node IP on the inter-node NIC}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"            # REAL GPUs per node (==> TP=EP=this, intra-node)
NUM_NODES="${NUM_NODES:-2}"                    # node count (==> DP=this, cross-node)
RAY_PORT="${RAY_PORT:-6379}"                   # Ray GCS port (head)
DASH_PORT="${DASH_PORT:-8265}"                 # Ray dashboard / job-submit port (head)

MODEL_NAME="${MODEL_NAME:-GLM-5.2_5layer}"
DSA_BACKEND="${DSA_BACKEND:-megatron-bridge}"  # unfused; use "slime" for fused TileLang
LORA_RANK="${LORA_RANK:-16}"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"               # == number of train steps
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"           # keep ~NUM_ROLLOUT/SAVE_INTERVAL adapters (disk)
TASK="${TASK:-gsm8k}"                          # example dataset: gsm8k | dapo-math (see run_glm5_lora.py)
RESP_LEN="${RESP_LEN:-}"                        # optional --rollout-max-response-len; for dapo-math try 4096
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-off}"  # on -> --dapo-dynamic-sampling (real model only; toy resamples forever)
HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/${MODEL_NAME}}"  # LOCAL dir, not a repo id
DATA_DIR="${DATA_DIR:-/personal/datasets}"     # persistent (NOT /root, which is volatile)
HF_HOME="${HF_HOME:-/cluster-storage/models}"
JOB_ID="${JOB_ID:-glm5_lora_mn_$(date +%y%m%d-%H%M%S)}"

# ----- wandb (default ON) -----
#  WANDB=on      -> enable_wandb stays True (default), so run_glm5_lora.py auto-adds
#                   `--use-wandb --wandb-project miles-run_glm5_lora --wandb-group <run_id>
#                    --wandb-key <WANDB_API_KEY> --disable-wandb-random-suffix`
#                   via U.get_default_wandb_args. ONLINE mode (wandb default).
#                   ⚠ needs WANDB_API_KEY in the env; if absent miles SKIPS wandb (warns, no error).
#  WANDB=offline -> same, but logs locally only (adds --wandb-mode offline; no key needed to log).
#  WANDB=off     -> pass --no-enable-wandb (no wandb at all).
WANDB="${WANDB:-on}"
WANDB_API_KEY="${WANDB_API_KEY:-}"             # required for online logging (export before running)
WANDB_TEAM="${WANDB_TEAM:-}"                   # optional wandb entity/team -> --wandb-team
WANDB_PROJECT="${WANDB_PROJECT:-}"             # optional override (default auto = miles-run_glm5_lora)
WANDB_GROUP="${WANDB_GROUP:-}"                 # optional override (default auto = run_id)

# Inter-node NIC for NCCL/GLOO. Auto-detect the iface holding an IP in HEAD_IP's /24,
# else override with NCCL_IFNAME=<iface> (the validated run used bond0).
if [[ -z "${NCCL_IFNAME:-}" ]]; then
  HEAD_PREFIX="${HEAD_IP%.*}."
  # Try `ip` (one line per addr: "N: <ifc>  inet <ip>/.."); each subst is `|| true` so a
  # missing tool can't trip `set -e`. Fall back to `ifconfig` (block format: "<ifc>: ..."
  # header then "inet <ip>" line) — some containers ship only one of the two.
  NCCL_IFNAME="$(ip -o -4 addr show 2>/dev/null | awk -v p="$HEAD_PREFIX" '$4 ~ ("^" p){print $2; exit}' || true)"
  if [[ -z "$NCCL_IFNAME" ]]; then
    NCCL_IFNAME="$(ifconfig 2>/dev/null | awk -v p="$HEAD_PREFIX" '/^[a-zA-Z]/{ifc=$1} $0 ~ ("inet "p){sub(/:$/,"",ifc); print ifc; exit}' || true)"
  fi
  NCCL_IFNAME="${NCCL_IFNAME:-eth0}"
fi

case "$ROLE" in
  # --------------------------------------------------------------------------
  head)
    echo "[head] (re)starting Ray head on ${HEAD_IP}:${RAY_PORT} with ${GPUS_PER_NODE} GPUs"
    ray stop --force >/dev/null 2>&1 || true
    sleep 3
    ray start --head \
      --node-ip-address "${HEAD_IP}" --port "${RAY_PORT}" \
      --dashboard-host 0.0.0.0 --dashboard-port "${DASH_PORT}" \
      --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
    echo
    echo "[head] now run THIS on each of the other $((NUM_NODES - 1)) node(s):"
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} bash scripts/run_glm5_lora_multinode.sh worker"
    echo "[head] waiting for ${NUM_NODES} nodes to join (Ctrl-C to stop waiting)..."
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    while :; do
      n="$(ray status 2>/dev/null | grep -cE '^ 1 node_' || true)"
      echo "[head]   nodes joined: ${n}/${NUM_NODES}"
      [[ "${n}" -ge "${NUM_NODES}" ]] && break
      sleep 5
    done
    echo "[head] cluster ready (${NUM_NODES} nodes). Next, on the head node run:"
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} MODEL_NAME=${MODEL_NAME} bash scripts/run_glm5_lora_multinode.sh launch"
    ;;

  # --------------------------------------------------------------------------
  worker)
    echo "[worker] (re)joining Ray head ${HEAD_IP}:${RAY_PORT} with ${GPUS_PER_NODE} GPUs"
    ray stop --force >/dev/null 2>&1 || true
    sleep 3
    ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}"
    echo "[worker] joined. (Verify on head: RAY_ADDRESS=${HEAD_IP}:${RAY_PORT} ray status)"
    ;;

  # --------------------------------------------------------------------------
  launch)
    echo "[launch] submitting ${MODEL_NAME} LoRA RL: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPU, backend=${DSA_BACKEND}, steps=${NUM_ROLLOUT}, job=${JOB_ID}"
    echo "[launch] NCCL/GLOO iface=${NCCL_IFNAME}"
    export HF_HOME PYTHONUNBUFFERED=1
    export NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" GLOO_SOCKET_IFNAME="${NCCL_IFNAME}"
    export MILES_SCRIPT_EXTERNAL_RAY=1                       # reuse the manually-formed cluster
    export RAY_ADDRESS="http://${HEAD_IP}:${DASH_PORT}"      # job-submit endpoint (dashboard)
    export MILES_RAY_SUBMIT_NO_WAIT=1                        # detached submit (survives WS drops)
    export MILES_RAY_SUBMISSION_ID="${JOB_ID}"

    # ----- wandb wiring (see knobs above) -----
    # WANDB_SCRIPT_FLAG: a run_glm5_lora.py (typer) flag. WANDB_EXTRA: miles/train.py
    # passthrough args that ride inside --extra-args. The launcher itself auto-adds
    # --use-wandb/--wandb-project miles-run_glm5_lora/--wandb-group <run_id>/--wandb-key
    # (from WANDB_API_KEY)/--disable-wandb-random-suffix when enable_wandb stays True;
    # extra_args is appended AFTER those, so any override here wins (argparse last-wins).
    WANDB_SCRIPT_FLAG=""
    WANDB_EXTRA=""
    if [[ "$WANDB" == "off" ]]; then
      WANDB_SCRIPT_FLAG="--no-enable-wandb"
      echo "[launch] wandb: OFF"
    else
      export WANDB_API_KEY                                  # get_default_wandb_args reads os.environ
      [[ -z "$WANDB_API_KEY" ]] && echo "[launch] wandb: WARN — WANDB_API_KEY empty -> miles SKIPS wandb (export it to log)."
      [[ "$WANDB" == "offline" ]] && WANDB_EXTRA+=" --wandb-mode offline"
      [[ -n "$WANDB_TEAM"    ]] && WANDB_EXTRA+=" --wandb-team ${WANDB_TEAM}"
      [[ -n "$WANDB_PROJECT" ]] && WANDB_EXTRA+=" --wandb-project ${WANDB_PROJECT}"
      [[ -n "$WANDB_GROUP"   ]] && WANDB_EXTRA+=" --wandb-group ${WANDB_GROUP}"
      echo "[launch] wandb: ${WANDB} (project auto=miles-run_glm5_lora unless overridden; key ${WANDB_API_KEY:+SET}${WANDB_API_KEY:-MISSING})"
    fi

    # ----- task / dataset flags (gsm8k | dapo-math; see run_glm5_lora.py) -----
    TASK_FLAGS="--task ${TASK}"
    [[ -n "$RESP_LEN" ]] && TASK_FLAGS+=" --rollout-max-response-len ${RESP_LEN}"
    [[ "$DAPO_DYNAMIC_SAMPLING" == "on" ]] && TASK_FLAGS+=" --dapo-dynamic-sampling"
    echo "[launch] task=${TASK}${RESP_LEN:+ resp_len=${RESP_LEN}} dapo_dynamic_sampling=${DAPO_DYNAMIC_SAMPLING}"

    # NOTE: --num-gpus-per-node $GPUS_PER_NODE appears BOTH as the run_glm5_lora.py flag
    # (drives TP=EP via _get_parallel_config + --actor-num-gpus-per-node) AND inside
    # --extra-args (sets the miles/train.py value the rollout addr-allocator reads). They
    # must agree with the REAL per-node GPU count — see the header's CRITICAL note.
    python scripts/run_glm5_lora.py train \
      --model-name "${MODEL_NAME}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --dsa-attention-backend "${DSA_BACKEND}" \
      --lora-rank "${LORA_RANK}" \
      --num-gpus-per-node "${GPUS_PER_NODE}" \
      --num-rollout "${NUM_ROLLOUT}" \
      --data-dir "${DATA_DIR}" \
      ${TASK_FLAGS} \
      ${WANDB_SCRIPT_FLAG} \
      --extra-args "--actor-num-nodes ${NUM_NODES} --num-gpus-per-node ${GPUS_PER_NODE} --save-interval ${SAVE_INTERVAL}${WANDB_EXTRA}"

    echo
    echo "[launch] submitted job '${JOB_ID}'. Monitor with:"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job status ${JOB_ID}"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job logs   ${JOB_ID} --follow"
    echo "[launch] verify topology in the logs: world_size=$((GPUS_PER_NODE * NUM_NODES)), actor-num-nodes=${NUM_NODES}, TP=${GPUS_PER_NODE}."
    ;;
esac
