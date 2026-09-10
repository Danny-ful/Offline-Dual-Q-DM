#!/usr/bin/env bash
set -euo pipefail

# --- 1) Optional mount wait (for startup scripts) ---
WAIT_FOR_MOUNT_SECONDS="${WAIT_FOR_MOUNT_SECONDS:-15}"
sleep "$WAIT_FOR_MOUNT_SECONDS"

# --- 2) Project bootstrap ---
PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM}"
cd "$PROJECT_ROOT"

# --- 3) Conda initialization ---
CONDA_PROFILE="${CONDA_PROFILE:-/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh}"
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

# --- 4) Train all four dynamics ensembles sequentially ---
# Example: DATASET_VARIANT=medium_expert-v2 bash scripts/run_dynamic.sh
# Extra arguments are forwarded to every run, e.g. dyn.epochs=100 seed=1.
DATASET_VARIANT="${DATASET_VARIANT:-full_replay-v2}"
# ENV_NAMES=(ant cheetah hopper walker)
# DATASET_PREFIXES=(ant halfcheetah hopper walker2d)
ENV_NAMES=(hopper)
DATASET_PREFIXES=(hopper)
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
  EXPERT_PATHS+=("$(find_dataset experts "${ENV_NAMES[$i]}" "${DATASET_PREFIXES[$i]}" .pkl)")
  SUPPLEMENT_PATHS+=("$(find_dataset supplement "${ENV_NAMES[$i]}" "${DATASET_PREFIXES[$i]}" "_${DATASET_VARIANT}.pkl")")
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
