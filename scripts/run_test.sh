#!/usr/bin/env bash
set -euo pipefail

# --- 1) Optional mount wait (for startup scripts) ---
WAIT_FOR_MOUNT_SECONDS="${WAIT_FOR_MOUNT_SECONDS:-15}"
sleep "$WAIT_FOR_MOUNT_SECONDS"

# --- 2) User/home/project bootstrap ---
export USER=ubuntu
export HOME=/home/ubuntu
PROJECT_ROOT="/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM"
cd "$PROJECT_ROOT"

# --- 3) Conda initialization ---
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

# MuJoCo 2.1 + mujoco_py runtime library path.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ubuntu/.mujoco/mujoco210/bin"

# Prefer active env's python, fall back to python3.
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No python executable found in PATH"
  exit 1
fi

# --- Stable offline IQ baseline WITHOUT dynamics penalty (targeting ~1.5k peak on Hopper) ---
# Prerequisite: matching files exist in both experts/ and supplement/.
ENV_NAME="${ENV_NAME:-hopper}"
ENV_DEMO="${ENV_DEMO:-Hopper-v2_d4rl.pkl}"
SEED="${SEED:-0}"
EXPERT_DEMOS="${EXPERT_DEMOS:-1}"
LEARN_STEPS="${LEARN_STEPS:-500000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_EPS="${EVAL_EPS:-20}"
EXP_NAME="${EXP_NAME:-offline_iq_hopper_no_penalty_1p5k_v1}"
PROJECT_NAME="${PROJECT_NAME:-Offline-Dual-Q-DM}"

"$PYTHON_BIN" train_iq.py \
  env="${ENV_NAME}" \
  agent=sac \
  offline=True \
  env.demo="${ENV_DEMO}" \
  expert.demos="${EXPERT_DEMOS}" \
  expert.subsample_freq=1 \
  method.loss=value \
  method.constrain=True \
  method.grad_pen=False \
  method.lambda_gp=0.1 \
  train.batch=256 \
  train.use_target=True \
  train.soft_update=True \
  agent.critic_lr=3e-4 \
  agent.critic_tau=0.005 \
  agent.actor_lr=3e-5 \
  agent.actor_update_frequency=1 \
  num_actor_updates=1 \
  agent.init_temp=1.0 \
  agent.learn_temp=False \
  cliptarget=False \
  penalty=3 \
  left=-2.0 \
  right=1.0 \
  method.regularize=False \
  method.chi=False \
  method.alpha=0.5 \
  actor_expert_offline=False \
  schedular=False \
  cuda_deterministic=True \
  env.learn_steps="${LEARN_STEPS}" \
  env.eval_interval="${EVAL_INTERVAL}" \
  eval.eps="${EVAL_EPS}" \
  project_name="${PROJECT_NAME}" \
  exp_name="${EXP_NAME}" \
  seed="${SEED}" \
  value_ratio=2.0 \
  actor_bc=False \
  actor_bc_coef=0.5 \
  method.penalty_auto=True \
  method.penalty_target=5 \
  method.penalty_lr=1e-2 \
  method.penalty=True \
  method.penalty_N=5 \
  method.dynamics_ckpt=dynamics/Hopper-v2/ensemble_5.pt
  # q_net._target_=agent.sac_models.DoubleQCritic