#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts_recoil/run_recoil_critic.sh
# Optional environment variables:
#   WANDB_ENTITY=wenqilaid-nanjing-university
#   WANDB_PROJECT=Offline-Dual-Q-DM
#   NUM_AGENTS=3
#   SWEEP_CONFIG=scripts_recoil/wandb_sweep_recoil_critic.yaml
#   SWEEP_LOG=scripts_recoil/sweep_ids_critic.log
#   EXISTING_SWEEP_ID=           # set non-empty to skip creating a new sweep
#   AUTO_REUSE_SWEEP=1           # if config unchanged, reuse last sweep_id
#   SWEEP_STATE_FILE=scripts_recoil/.sweep_state_critic.env
#   AGENT_LAUNCH_STAGGER_SECONDS=1  # delay between agent launches

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

WANDB_ENTITY="${WANDB_ENTITY:-wenqilaid-nanjing-university}"
WANDB_PROJECT="${WANDB_PROJECT:-Offline-Dual-Q-DM}"
NUM_AGENTS="${NUM_AGENTS:-3}"
SWEEP_CONFIG="${SWEEP_CONFIG:-scripts_recoil/wandb_sweep_recoil_critic.yaml}"
SWEEP_LOG="${SWEEP_LOG:-scripts_recoil/sweep_ids_critic.log}"
EXISTING_SWEEP_ID="${EXISTING_SWEEP_ID:-}"
AUTO_REUSE_SWEEP="${AUTO_REUSE_SWEEP:-1}"
SWEEP_STATE_FILE="${SWEEP_STATE_FILE:-scripts_recoil/.sweep_state_critic.env}"
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

if command -v sha256sum >/dev/null 2>&1; then
  CONFIG_HASH="$(sha256sum "$SWEEP_CONFIG_RESOLVED" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  CONFIG_HASH="$(shasum -a 256 "$SWEEP_CONFIG_RESOLVED" | awk '{print $1}')"
else
  echo "Neither sha256sum nor shasum is available to hash sweep config."
  exit 1
fi

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb command not found. Install wandb in the active environment first."
  exit 1
fi

STATE_SWEEP_CONFIG=""
STATE_CONFIG_HASH=""
STATE_WANDB_ENTITY=""
STATE_WANDB_PROJECT=""
STATE_SWEEP_ID=""
STATE_AGENT_TARGET=""
if [ -f "$SWEEP_STATE_FILE" ]; then
  # shellcheck source=/dev/null
  source "$SWEEP_STATE_FILE"
fi

if [ -n "$EXISTING_SWEEP_ID" ]; then
  echo "Reusing existing sweep ID from EXISTING_SWEEP_ID: ${EXISTING_SWEEP_ID}"
  SWEEP_ID="$EXISTING_SWEEP_ID"
  if [ -n "$WANDB_ENTITY" ]; then
    AGENT_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
  else
    echo "EXISTING_SWEEP_ID is set but WANDB_ENTITY is empty."
    echo "Set WANDB_ENTITY, e.g. WANDB_ENTITY=wenqilaid-nanjing-university"
    exit 1
  fi
elif [ "$AUTO_REUSE_SWEEP" = "1" ] \
  && [ -n "${STATE_SWEEP_ID:-}" ] \
  && [ -n "${STATE_AGENT_TARGET:-}" ] \
  && [ "${STATE_SWEEP_CONFIG:-}" = "$SWEEP_CONFIG_RESOLVED" ] \
  && [ "${STATE_CONFIG_HASH:-}" = "$CONFIG_HASH" ] \
  && [ "${STATE_WANDB_ENTITY:-}" = "$WANDB_ENTITY" ] \
  && [ "${STATE_WANDB_PROJECT:-}" = "$WANDB_PROJECT" ]; then
  SWEEP_ID="$STATE_SWEEP_ID"
  AGENT_TARGET="$STATE_AGENT_TARGET"
  echo "Sweep config unchanged; reusing previous sweep ID: $SWEEP_ID"
else
  echo "Creating new sweep from ${SWEEP_CONFIG} ..."
  if [ "$AUTO_REUSE_SWEEP" = "1" ] && [ -n "${STATE_SWEEP_ID:-}" ]; then
    echo "Detected config or project/entity change, auto-rotating sweep_id."
  fi
  if [ -n "$WANDB_ENTITY" ]; then
    SWEEP_OUT="$(wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)"
  else
    SWEEP_OUT="$(wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)"
  fi
  echo "$SWEEP_OUT"

  # Strip ANSI escape sequences before parsing in case wandb colors output.
  SWEEP_OUT_CLEAN="$(printf '%s\n' "$SWEEP_OUT" | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g')"

  # Prefer parsing the explicit sweep-id line first.
  SWEEP_ID="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*Creating sweep with ID: \([A-Za-z0-9_-]\+\).*/\1/p' | tail -n 1)"

  # Fallback: parse full agent path and extract trailing id.
  if [ -z "$SWEEP_ID" ]; then
    AGENT_PATH="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*wandb agent \(.*\)$/\1/p' | tail -n 1)"
    if [ -n "$AGENT_PATH" ]; then
      SWEEP_ID="${AGENT_PATH##*/}"
    fi
  fi

  if [ -z "$SWEEP_ID" ]; then
    echo "Failed to parse sweep ID from wandb output."
    echo "You can still copy the 'wandb agent ...' line above and run it manually."
    exit 1
  fi

  if [ -n "$WANDB_ENTITY" ]; then
    AGENT_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
  else
    AGENT_TARGET="$(printf '%s\n' "$SWEEP_OUT_CLEAN" | sed -n 's/.*wandb agent \(.*\)$/\1/p' | tail -n 1)"
    if [ -z "$AGENT_TARGET" ]; then
      echo "Failed to parse full wandb agent target."
      echo "Please run manually: wandb agent <entity>/${WANDB_PROJECT}/${SWEEP_ID}"
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
  printf 'STATE_CONFIG_HASH=%q\n' "$CONFIG_HASH"
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
