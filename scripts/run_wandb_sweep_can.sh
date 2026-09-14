#!/usr/bin/env bash
set -euo pipefail

# CAN WandB Sweep Script
# Usage:
#   bash scripts/run_wandb_sweep_can.sh
# Optional environment variables:
#   WANDB_ENTITY=wenqilaid-nanjing-university
#   WANDB_PROJECT=robosuite
#   NUM_AGENTS=3
#   SWEEP_CONFIG=scripts/wandb_sweep_can.yaml

export USER=ubuntu
export HOME=/home/ubuntu
PROJECT_ROOT="/home/ubuntu/laiwenqi/projects/Offline Dual Q-DM"
cd "$PROJECT_ROOT"

# Find and source conda
CONDA_PROFILE="/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh"
ALT_CONDA_PROFILE="/home/ubuntu/anaconda3/etc/profile.d/conda.sh"
MINI_CONDA_PROFILE="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PROFILE" ]; then
  source "$CONDA_PROFILE"
elif [ -f "$ALT_CONDA_PROFILE" ]; then
  source "$ALT_CONDA_PROFILE"
elif [ -f "$MINI_CONDA_PROFILE" ]; then
  source "$MINI_CONDA_PROFILE"
else
  echo "conda.sh not found."
  exit 1
fi
conda activate IQ

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ubuntu/.mujoco/mujoco210/bin"

# Refuse silent CPU fallback when the cloud GPU is not ready.
# shellcheck source=scripts/gpu_preflight.sh
source scripts/gpu_preflight.sh
gpu_preflight "${PYTHON_BIN:-python}"

WANDB_ENTITY="${WANDB_ENTITY:-wenqilaid-nanjing-university}"
WANDB_PROJECT="${WANDB_PROJECT:-robosuite}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SWEEP_CONFIG="${SWEEP_CONFIG:-scripts/wandb_sweep_can.yaml}"
SWEEP_LOG="${SWEEP_LOG:-scripts/sweep_ids_can.log}"
EXISTING_SWEEP_ID="${EXISTING_SWEEP_ID:-}"
AUTO_REUSE_SWEEP="${AUTO_REUSE_SWEEP:-1}"
SWEEP_STATE_FILE="${SWEEP_STATE_FILE:-scripts/.sweep_state_can.env}"
AGENT_LAUNCH_STAGGER_SECONDS="${AGENT_LAUNCH_STAGGER_SECONDS:-1}"

if [ ! -f "$SWEEP_CONFIG" ]; then
  echo "Sweep config not found: $SWEEP_CONFIG"
  exit 1
fi

if command -v realpath >/dev/null 2>&1; then
  SWEEP_CONFIG_RESOLVED="$(realpath "$SWEEP_CONFIG")"
else
  SWEEP_CONFIG_RESOLVED="$SWEEP_CONFIG"
fi

if [ -n "$EXISTING_SWEEP_ID" ]; then
  echo "Reusing existing sweep ID from EXISTING_SWEEP_ID: ${EXISTING_SWEEP_ID}"
  SWEEP_ID="$EXISTING_SWEEP_ID"
  if [ -n "$WANDB_ENTITY" ]; then
    AGENT_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
  else
    echo "EXISTING_SWEEP_ID is set but WANDB_ENTITY is empty."
    exit 1
  fi
else
  echo "Creating sweep from ${SWEEP_CONFIG} ..."
  if [ -n "$WANDB_ENTITY" ]; then
    SWEEP_OUT="$(wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)"
  else
    SWEEP_OUT="$(wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)"
  fi
  echo "$SWEEP_OUT"

  SWEEP_OUT_CLEAN="$(printf '%s\n' "$SWEEP_OUT" | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g')"

  SWEEP_ID="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*Creating sweep with ID: \([A-Za-z0-9_-]\+\).*/\1/p' | tail -n 1)"

  if [ -z "$SWEEP_ID" ]; then
    AGENT_PATH="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*wandb agent \(.*\)$/\1/p' | tail -n 1)"
    if [ -n "$AGENT_PATH" ]; then
      SWEEP_ID="${AGENT_PATH##*/}"
    fi
  fi

  if [ -z "$SWEEP_ID" ]; then
    echo "Failed to parse sweep ID from wandb output."
    exit 1
  fi

  if [ -n "$WANDB_ENTITY" ]; then
    AGENT_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
  else
    AGENT_TARGET="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*wandb agent \(.*\)$/\1/p' | tail -n 1)"
    if [ -z "$AGENT_TARGET" ]; then
      echo "Failed to parse full wandb agent target."
      exit 1
    fi
  fi
fi

mkdir -p "$(dirname "$SWEEP_LOG")"
{
  printf '%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$AGENT_TARGET"
} >> "$SWEEP_LOG"

mkdir -p "$(dirname "$SWEEP_STATE_FILE")"
{
  printf 'STATE_SWEEP_CONFIG=%q\n' "$SWEEP_CONFIG_RESOLVED"
  printf 'STATE_WANDB_ENTITY=%q\n' "$WANDB_ENTITY"
  printf 'STATE_WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
  printf 'STATE_SWEEP_ID=%q\n' "$SWEEP_ID"
  printf 'STATE_AGENT_TARGET=%q\n' "$AGENT_TARGET"
} > "$SWEEP_STATE_FILE"

echo "Parsed sweep ID: $SWEEP_ID"
echo "Agent target: $AGENT_TARGET"
echo "Recorded sweep target to: $SWEEP_LOG"
echo "Recorded sweep state to: $SWEEP_STATE_FILE"
echo "Launching ${NUM_AGENTS} agent(s)..."

i=1
while [ "$i" -le "$NUM_AGENTS" ]; do
  wandb agent "$AGENT_TARGET" &
  if [ "$i" -lt "$NUM_AGENTS" ]; then
    sleep "$AGENT_LAUNCH_STAGGER_SECONDS"
  fi
  i=$((i + 1))
done

wait
