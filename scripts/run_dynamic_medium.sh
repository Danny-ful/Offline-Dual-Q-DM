#!/usr/bin/env bash
set -euo pipefail

# --- 1) Project bootstrap ---
PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/shengyifei/lwq/Offline-Dual-Q-DM}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/shengyifei/lwq}"
EXPERT_DIR="${EXPERT_DIR:-$DATA_ROOT/experts}"
SUPPLEMENT_DIR="${SUPPLEMENT_DIR:-$DATA_ROOT/supplement}"
cd "$PROJECT_ROOT"

# --- 2) Conda initialization ---
CONDA_PROFILE="${CONDA_PROFILE:-/home/ubuntu/shengyifei/anaconda3/etc/profile.d/conda.sh}"
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

# Use the same pinned IQ environment as the MuJoCo sweep scripts.
IQ_ENV_PATH="${IQ_ENV_PATH:-/home/ubuntu/shengyifei/conda_cache/envs/IQ}"
if [ -d "$IQ_ENV_PATH" ]; then
  conda activate "$IQ_ENV_PATH"
else
  echo "Error: IQ environment not found at $IQ_ENV_PATH"
  echo "Available environments:"
  conda info --envs
  exit 1
fi

# Match the MuJoCo runtime used by the walker sweep.
#MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ubuntu/shengyifei/lwq/runtime/mujoco210}"
#export MUJOCO_PY_MUJOCO_PATH
# Keep mujoco_py on the CPU/OSMesa extension path so it does not rebuild a
# GPU extension inside a worker with a different runtime environment.
#export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
#unset MUJOCO_PY_MUJOCO_PATH
#export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#export LD_LIBRARY_PATH="/home/ubuntu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# --- MuJoCo 2.1 runtime ---
# --- MuJoCo / OSMesa runtime ---
MUJOCO_ROOT="/home/ubuntu/shengyifei/lwq/runtime/mujoco210"
OSMESA_ROOT="/home/ubuntu/shengyifei/lwq/runtime/osmesa_focal_v1"

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT"
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"

# Runtime dynamic libraries
export LD_LIBRARY_PATH="$MUJOCO_ROOT/bin:$OSMESA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Headers, only needed if mujoco_py ever rebuilds
export CPATH="$OSMESA_ROOT/include${CPATH:+:$CPATH}"

# Libraries, only needed for compilation/linking
export LIBRARY_PATH="$OSMESA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# Prefer active env's python, fall back to python3.
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# --- 3) Refuse silent CPU fallback when the cloud GPU is not ready. ---
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No python executable found in PATH"
  exit 1
fi

# shellcheck source=scripts/gpu_preflight.sh
source scripts/gpu_preflight.sh
gpu_preflight "$PYTHON_BIN"
# --- 3.5) MuJoCo runtime preflight ---
echo "===== mujoco_py existing extensions ====="

MUJOCO_PY_DIR="/home/ubuntu/shengyifei/conda_cache/envs/IQ/lib/python3.10/site-packages/mujoco_py"

find "$MUJOCO_PY_DIR/generated" \
  -type f -name 'cymj*.so' -print || true

echo "===== ldd cymj ====="

find "$MUJOCO_PY_DIR/generated" \
  -type f -name 'cymj*.so' \
  -exec sh -c '
    echo "---- $1 ----"
    ldd "$1"
  ' _ {} \; || true

echo "===== OSMesa runtime ====="

ldconfig -p 2>/dev/null | grep -i osmesa || true

find /usr /lib /home/ubuntu/shengyifei/lwq/runtime \
  -name 'libOSMesa.so*' -print 2>/dev/null | head -30

echo "===== OSMesa header ====="

find /usr/include /home/ubuntu/shengyifei/lwq/runtime \
  -path '*/GL/osmesa.h' -print 2>/dev/null || true


# --- 4) Train all four dynamics ensembles sequentially ---
# Example: DATASET_VARIANT=medium-v2 bash scripts/run_dynamic_medium.sh
# Extra arguments are forwarded to every run, e.g. dyn.epochs=100 seed=1.
DATASET_VARIANT="${DATASET_VARIANT:-medium-v2}"
# ENV_NAMES=(ant cheetah hopper walker)
# DATASET_PREFIXES=(ant halfcheetah hopper walker2d)
ENV_NAMES=(ant halfcheetah hopper walker2d)
DATASET_PREFIXES=(ant halfcheetah hopper walker2d)
EXPERT_PATHS=()
SUPPLEMENT_PATHS=()

# Prefer the config name, accepting the D4RL name as an alternative.
find_dataset() {
  local directory="$1" env_name="$2" dataset_prefix="$3" suffix="$4"
  local primary="${directory}/${env_name}${suffix}"
  local alternate="${directory}/${dataset_prefix}${suffix}"
  if [[ -f "$primary" ]]; then
    printf '%s\n' "$primary"
  elif [[ -f "$alternate" ]]; then
    printf '%s\n' "$alternate"
  else
    printf 'Missing dataset for %s. Checked: %s and %s\n' \
      "$env_name" "$primary" "$alternate" >&2
    return 1
  fi
}

# Check every input before spending time training the first environment.
for i in "${!ENV_NAMES[@]}"; do
EXPERT_PATHS+=("$(find_dataset "$EXPERT_DIR" "${ENV_NAMES[$i]}" "${DATASET_PREFIXES[$i]}" .pkl)")
SUPPLEMENT_PATHS+=("$(find_dataset "$SUPPLEMENT_DIR" "${ENV_NAMES[$i]}" "${DATASET_PREFIXES[$i]}" "_${DATASET_VARIANT}.pkl")")
done

for i in "${!ENV_NAMES[@]}"; do
  demo="${SUPPLEMENT_PATHS[$i]##*/}"
  echo "=== [$((i + 1))/${#ENV_NAMES[@]}] Training dynamics: ${ENV_NAMES[$i]} ==="
  echo "Expert: ${EXPERT_PATHS[$i]} | Supplement: ${SUPPLEMENT_PATHS[$i]}"
  "$PYTHON_BIN" train_dynamics.py \
    "env=${ENV_NAMES[$i]}" \
    "env.expert_path=${EXPERT_PATHS[$i]}" \
    "env.supplement_path=${SUPPLEMENT_PATHS[$i]}" \
    "env.demo=$demo" \
    "$@"
done

echo "=== Finished dynamics training for ant, cheetah, hopper and walker ==="
