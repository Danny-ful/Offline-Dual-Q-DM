#!/usr/bin/env bash
set -euo pipefail

# Robosuite IQ-Learn Experiments - Expert (PH) + Supplement (MG)
# Usage:
#   bash scripts/run_robosuite_experiments.sh

export USER=ubuntu
export HOME=/home/ubuntu
PROJECT_ROOT="/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM"
cd "$PROJECT_ROOT"

# Find and source conda
CONDA_PROFILE="/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh"
ALT_CONDA_PROFILE="/home/ubuntu/anaconda3/etc/profile.d/conda.sh"
MINI_CONDA_PROFILE="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PROFILE" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_PROFILE"
elif [ -f "$ALT_CONDA_PROFILE" ]; then
  # shellcheck source=/dev/null
  source "$ALT_CONDA_PROFILE"
elif [ -f "$MINI_CONDA_PROFILE" ]; then
  # shellcheck source=/dev/null
  source "$MINI_CONDA_PROFILE"
else
  echo "conda.sh not found. Checked:"
  echo "  $CONDA_PROFILE"
  echo "  $ALT_CONDA_PROFILE"
  echo "  $MINI_CONDA_PROFILE"
  exit 1
fi
conda activate IQ

# Set up MuJoCo (needed for robosuite)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ubuntu/.mujoco/mujoco210/bin"

# Refuse silent CPU fallback when the cloud GPU is not ready.
# shellcheck source=scripts/gpu_preflight.sh
source scripts/gpu_preflight.sh
gpu_preflight "${PYTHON_BIN:-python}"

# =====================================================
# Experiment: Lift task with PH (expert) + MG (supplement)
# =====================================================
echo "=========================================="
echo "Robosuite IQ-Learn Training"
echo "Expert Dataset: Lift-PH (all trajectories)"
echo "Supplement Dataset: Lift-MG sparse (all trajectories)"
echo "=========================================="

"${PYTHON_BIN:-python}" train_iq_robosuite.py \
    project_name=robosuite \
    robosuite.task=lift \
    robosuite.expert_dataset_type=ph \
    robosuite.expert_hdf5_type=low_dim \
    robosuite.expert_trajs=null \
    robosuite.supplement_dataset_type=mg \
    robosuite.supplement_hdf5_type=low_dim_sparse \
    robosuite.supplement_trajs=null \
    robosuite.use_supplement=true \
    robosuite.learn_steps=100000 \
    robosuite.eval_interval=5000 \
    robosuite.eval_episodes=100 \
    robosuite.eval_on_env=true \
    eval.stochastic=false \
    agent.learn_temp=false \
    method.synthetic_constrain=true \
    method.uncertainty=true \
    'method.dynamics_ckpt=/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM/dynamics/robosuite/lift/ph_low_dim__mg_low_dim_sparse/ensemble_5.pt' \
    agent.actor_lr=3e-05 \
    agent.critic_lr=0.0001 \
    agent.init_temp=0.1 \
    method.alpha=0.5 \
    method.penalty_coef=0.5 \
    method.penalty_target=1 \
    method.synthetic_coef=3 \
    method.synthetic_warmup_steps=1 \
    seed=0 \
    train.batch=256

echo ""
echo "Training completed!"
echo "The exact run-specific log directory is printed by the training process."
