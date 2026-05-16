#!/usr/bin/env bash
set -euo pipefail

# Single-trajectory BC stable baseline runner.
#
# Goal:
# - Train BC with only ONE expert trajectory.
# - Keep training/evaluation stable and reproducible for baseline usage.
#
# Optional environment variables:
#   WAIT_FOR_MOUNT_SECONDS=15
#   ENV_NAME=hopper
#   ENV_DEMO=Hopper-v2_d4rl.pkl
#   SEED=0
#   LEARN_STEPS=300000
#   EVAL_INTERVAL=2500
#   EVAL_EPS=50
#   BATCH_SIZE=512
#   ACTOR_LR=3e-5
#   EARLY_STOP_PATIENCE=10
#   EARLY_STOP_MIN_DELTA=5.0
#   EXP_NAME=bc_1traj_stable_baseline
#   PROJECT_NAME=Offline-Dual-Q-DM

WAIT_FOR_MOUNT_SECONDS="${WAIT_FOR_MOUNT_SECONDS:-15}"
sleep "$WAIT_FOR_MOUNT_SECONDS"

export USER=ubuntu
export HOME=/home/ubuntu
PROJECT_ROOT="/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM"
cd "$PROJECT_ROOT"

CONDA_PROFILE="/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh"
ALT_CONDA_PROFILE="/home/ubuntu/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_PROFILE"
elif [ -f "$ALT_CONDA_PROFILE" ]; then
  # shellcheck source=/dev/null
  source "$ALT_CONDA_PROFILE"
else
  echo "conda.sh not found. Checked:"
  echo "  $CONDA_PROFILE"
  echo "  $ALT_CONDA_PROFILE"
  exit 1
fi
conda activate IQ

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ubuntu/.mujoco/mujoco210/bin"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No python executable found in PATH"
  exit 1
fi

ENV_NAME="${ENV_NAME:-hopper}"
ENV_DEMO="${ENV_DEMO:-Hopper-v2_d4rl.pkl}"
SEED="${SEED:-0}"
LEARN_STEPS="${LEARN_STEPS:-300000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
EVAL_EPS="${EVAL_EPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-5.0}"
EXP_NAME="${EXP_NAME:-bc_1traj_stable_baseline}"
PROJECT_NAME="${PROJECT_NAME:-Offline-Dual-Q-DM}"

echo "Running stable BC baseline with one expert trajectory..."
echo "env=${ENV_NAME}, demo=${ENV_DEMO}, seed=${SEED}"
echo "learn_steps=${LEARN_STEPS}, eval_interval=${EVAL_INTERVAL}, eval_eps=${EVAL_EPS}"
echo "batch=${BATCH_SIZE}, actor_lr=${ACTOR_LR}, exp_name=${EXP_NAME}"
echo "early_stop_patience=${EARLY_STOP_PATIENCE}, early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"

"$PYTHON_BIN" train_iq.py \
  env="${ENV_NAME}" \
  agent=sac \
  method=bc \
  method.loss=nll \
  offline=True \
  env.demo="${ENV_DEMO}" \
  expert.demos=1 \
  expert.subsample_freq=1 \
  env.learn_steps="${LEARN_STEPS}" \
  env.eval_interval="${EVAL_INTERVAL}" \
  eval.eps="${EVAL_EPS}" \
  train.batch="${BATCH_SIZE}" \
  agent.actor_lr="${ACTOR_LR}" \
  schedular=False \
  cuda_deterministic=True \
  method.early_stop=True \
  method.early_stop_patience="${EARLY_STOP_PATIENCE}" \
  method.early_stop_min_delta="${EARLY_STOP_MIN_DELTA}" \
  project_name="${PROJECT_NAME}" \
  exp_name="${EXP_NAME}" \
  seed="${SEED}"
