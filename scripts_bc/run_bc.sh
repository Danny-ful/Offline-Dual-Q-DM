#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts_bc/run_bc.sh
#
# Runs a single BC training job with the default Hopper NLL parameters below.
# Override any value with environment variables, for example:
#   SEED=1 EXP_NAME=sweep_hopper_bc_nll_seed1 bash scripts_bc/run_bc.sh

WAIT_FOR_MOUNT_SECONDS="${WAIT_FOR_MOUNT_SECONDS:-0}"
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
EXPERT_DEMOS="${EXPERT_DEMOS:-1}"
EXPERT_SUBSAMPLE_FREQ="${EXPERT_SUBSAMPLE_FREQ:-1}"
PROJECT_NAME="${PROJECT_NAME:-Offline-Dual-Q-DM}"
EXP_NAME="${EXP_NAME:-sweep_hopper_bc_nll}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARN_STEPS="${LEARN_STEPS:-50000}"

echo "Running one BC job..."
echo "env=${ENV_NAME}, demo=${ENV_DEMO}, seed=${SEED}"
echo "expert.demos=${EXPERT_DEMOS}, expert.subsample_freq=${EXPERT_SUBSAMPLE_FREQ}"
echo "loss=nll, learn_steps=${LEARN_STEPS}, batch=${BATCH_SIZE}, actor_lr=${ACTOR_LR}, exp_name=${EXP_NAME}"

"$PYTHON_BIN" train_iq.py \
  env="${ENV_NAME}" \
  agent=sac \
  method=bc \
  seed="${SEED}" \
  method.loss=nll \
  env.demo="${ENV_DEMO}" \
  env.learn_steps="${LEARN_STEPS}" \
  expert.demos="${EXPERT_DEMOS}" \
  expert.subsample_freq="${EXPERT_SUBSAMPLE_FREQ}" \
  project_name="${PROJECT_NAME}" \
  exp_name="${EXP_NAME}" \
  agent.actor_lr="${ACTOR_LR}" \
  train.batch="${BATCH_SIZE}"
