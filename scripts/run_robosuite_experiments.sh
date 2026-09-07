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

# =====================================================
# Experiment: Lift task with PH (expert) + MG (supplement)
# =====================================================
echo "=========================================="
echo "Robosuite IQ-Learn Training"
echo "Expert Dataset: Lift-PH (all trajectories)"
echo "Supplement Dataset: Lift-MG sparse (all trajectories)"
echo "=========================================="

python train_iq_robosuite.py \
    robosuite.task=lift \
    robosuite.expert_dataset_type=ph \
    robosuite.expert_hdf5_type=low_dim \
    robosuite.expert_trajs=null \
    robosuite.supplement_dataset_type=mg \
    robosuite.supplement_hdf5_type=low_dim_sparse \
    robosuite.supplement_trajs=null \
    robosuite.use_supplement=true \
    robosuite.learn_steps=1000000 \
    robosuite.eval_interval=5000 \
    robosuite.eval_episodes=10 \
    train.batch=256 \
    method.alpha=1.0 \
    project_name=robosuite \
    exp_name=lift_ph_expert_mg_sparse_supp \
    seed=0

echo ""
echo "Training completed!"
echo "The exact run-specific log directory is printed by the training process."
