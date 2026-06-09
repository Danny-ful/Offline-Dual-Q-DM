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

# --- Stable offline IQ baseline using train_iq_offline.py ---

"$PYTHON_BIN" -m wandb sync --sync-tensorboard logs/offline >/dev/null 2>&1 &
WANDB_SYNC_PID="$!"
cleanup() {
  if kill -0 "$WANDB_SYNC_PID" >/dev/null 2>&1; then
    kill "$WANDB_SYNC_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT


echo "=== Stage 1: Training Dynamics Model ==="

"$PYTHON_BIN" train_dynamics.py \
    env=hopper \
    env.demo=Hopper-v2_d4rl.pkl