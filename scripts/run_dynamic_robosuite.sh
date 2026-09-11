#!/usr/bin/env bash
set -euo pipefail

# Train Lift and Can sequentially on the server. Hydra overrides apply to both.
# TASKS=lift bash scripts/run_dynamic_robosuite.sh dyn.epochs=2
# bash scripts/run_dynamic_robosuite.sh robosuite.supplement_hdf5_type=low_dim_dense
PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM}"
cd "$PROJECT_ROOT"

# Use CONDA_ENV='' to keep the current Python environment.
CONDA_ENV="${CONDA_ENV-IQ}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  profiles=(
    "${CONDA_PROFILE:-/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh}"
    /home/ubuntu/anaconda3/etc/profile.d/conda.sh
    /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  )
  found=false
  for profile in "${profiles[@]}"; do
    if [[ -f "$profile" ]]; then
      source "$profile"
      found=true
      break
    fi
  done
  if [[ "$found" != true ]]; then
    echo "conda.sh not found; set CONDA_PROFILE or use CONDA_ENV='' with an active Python environment." >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
read -r -a tasks <<< "${TASKS:-lift can}"
for task in "${tasks[@]}"; do
  case "$task" in
    lift|can) ;;
    *) echo "Unsupported task: $task (use TASKS='lift can')" >&2; exit 1 ;;
  esac
done

# Check both datasets before training either model. This uses the real loader
# and trajectory split, but never instantiates a simulator or fits a model.
for task in "${tasks[@]}"; do
  "$PYTHON_BIN" train_dynamics_robosuite.py "$@" "robosuite.task=$task" dyn.check_only=true
done
for task in "${tasks[@]}"; do
  "$PYTHON_BIN" train_dynamics_robosuite.py "$@" "robosuite.task=$task" dyn.check_only=false
done
echo "Finished dynamics training for: ${tasks[*]}"
