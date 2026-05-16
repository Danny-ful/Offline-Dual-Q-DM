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

# --- Offline + supplement training ---
# Prerequisite: matching files exist in both experts/ and supplement/.
"$PYTHON_BIN" train_iq.py env=hopper agent=sac offline=True method.loss=value env.demo=Hopper-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 project_name=Offline-Dual-Q-DM exp_name=offline_supplement_hopper_value seed=0

"$PYTHON_BIN" train_iq.py env=hopper agent=sac offline=True method.loss=value_expert env.demo=Hopper-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 project_name=Offline-Dual-Q-DM exp_name=offline_supplement_hopper_expertonly seed=0

# Constrain ablation on the same dataset.
"$PYTHON_BIN" train_iq.py env=hopper agent=sac offline=True method.loss=value env.demo=Hopper-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 method.constrain=True project_name=Offline-Dual-Q-DM exp_name=offline_supplement_hopper_value_constrain seed=0

"$PYTHON_BIN" train_iq.py env=hopper agent=sac offline=True method.loss=value_expert env.demo=Hopper-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 method.constrain=True project_name=Offline-Dual-Q-DM exp_name=offline_supplement_hopper_expertonly_constrain seed=0

# Optional:
# "$PYTHON_BIN" train_iq.py env=cheetah agent=sac offline=True method.loss=value_expert env.demo=HalfCheetah-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 project_name=Offline-Dual-Q-DM exp_name=offline_supplement_cheetah seed=0
# "$PYTHON_BIN" train_iq.py env=walker agent=sac offline=True method.loss=value_expert env.demo=Walker2d-v2_d4rl.pkl expert.demos=25 expert.subsample_freq=1 project_name=Offline-Dual-Q-DM exp_name=offline_supplement_walker seed=0