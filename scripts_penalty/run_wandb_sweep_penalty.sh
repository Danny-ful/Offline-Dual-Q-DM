#!/usr/bin/env bash
set -euo pipefail

# One-click sweep launcher.
#
# Workflow:
#   1) Parse method.dynamics_ckpt / method.penalty_N / env=... / env.demo=... from
#      the sweep yaml.
#   2) If the dynamics ensemble checkpoint is missing (or FORCE_RETRAIN_DYN=1),
#      pretrain it once with train_dynamics.py. This is done ONCE per sweep
#      (all agents share the same frozen checkpoint).
#   3) Create the wandb sweep and launch NUM_AGENTS agents. Each agent will
#      internally spawn train_iq.py with hyperparameters drawn by wandb.
#
# Usage:
#   bash scripts_penalty/run_wandb_sweep_penalty.sh
#
# Optional environment variables:
#   WANDB_ENTITY=wenqilaid-nanjing-university
#   WANDB_PROJECT=Offline-Dual-Q-DM
#   NUM_AGENTS=3
#   SWEEP_CONFIG=scripts_penalty/wandb_sweep_penalty.yaml
#   SWEEP_LOG=scripts_penalty/sweep_ids.log
#   EXISTING_SWEEP_ID=           # set non-empty to skip creating a new sweep
#   AUTO_REUSE_SWEEP=1           # if config unchanged, reuse last sweep_id
#   SWEEP_STATE_FILE=scripts_penalty/.sweep_state.env
#   FORCE_RETRAIN_DYN=0         # 1 to re-train the dynamics ensemble
#   DYN_EPOCHS=100              # epochs for dynamics pretraining
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
SWEEP_CONFIG="${SWEEP_CONFIG:-scripts_penalty/wandb_sweep_penalty.yaml}"
SWEEP_LOG="${SWEEP_LOG:-scripts_penalty/sweep_ids.log}"
EXISTING_SWEEP_ID="${EXISTING_SWEEP_ID:-}"
AUTO_REUSE_SWEEP="${AUTO_REUSE_SWEEP:-1}"
SWEEP_STATE_FILE="${SWEEP_STATE_FILE:-scripts_penalty/.sweep_state.env}"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
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

# -------- Stage 1: Pretrain dynamics ensemble (once per sweep) --------
# Parse the key settings directly from the sweep yaml so this launcher stays in
# sync with whatever env / ckpt / ensemble size the sweep itself will use.
parse_cmd_val() {
  # $1: key name (regex-escaped dots), reads from $SWEEP_CONFIG
  sed -n "s/^[[:space:]]*-[[:space:]]*$1=\(.*\)$/\1/p" "$SWEEP_CONFIG" | tail -n 1
}

CKPT_PATH="$(parse_cmd_val 'method\.dynamics_ckpt' || true)"
SWEEP_N="$(parse_cmd_val 'method\.penalty_N' || true)"
SWEEP_ENV="$(parse_cmd_val 'env' || true)"
SWEEP_DEMO="$(parse_cmd_val 'env\.demo' || true)"

if [ -z "$CKPT_PATH" ]; then
  echo "Could not parse 'method.dynamics_ckpt=...' from $SWEEP_CONFIG."
  echo "Please add it to the yaml command section."
  exit 1
fi
if [ -z "$SWEEP_N" ]; then
  echo "Could not parse 'method.penalty_N=...' from $SWEEP_CONFIG."
  exit 1
fi
if [ -z "$SWEEP_ENV" ] || [ -z "$SWEEP_DEMO" ]; then
  echo "Could not parse 'env=...' or 'env.demo=...' from $SWEEP_CONFIG."
  exit 1
fi

FORCE_RETRAIN_DYN="${FORCE_RETRAIN_DYN:-0}"
DYN_EPOCHS="${DYN_EPOCHS:-100}"

if [ ! -f "$CKPT_PATH" ] || [ "$FORCE_RETRAIN_DYN" = "1" ]; then
  if [ "$FORCE_RETRAIN_DYN" = "1" ]; then
    echo "=> FORCE_RETRAIN_DYN=1, (re)training dynamics ensemble (N=$SWEEP_N, env=$SWEEP_ENV)..."
  else
    echo "=> Dynamics ensemble not found at $CKPT_PATH"
    echo "=> Training dynamics ensemble (N=$SWEEP_N, env=$SWEEP_ENV, demo=$SWEEP_DEMO)..."
  fi
  "$PYTHON_BIN" train_dynamics.py \
    env="$SWEEP_ENV" agent=sac offline=True \
    env.demo="$SWEEP_DEMO" expert.demos=1 expert.subsample_freq=1 \
    method=iq method.penalty_N="$SWEEP_N" \
    +dyn.epochs="$DYN_EPOCHS"
  if [ ! -f "$CKPT_PATH" ]; then
    echo "train_dynamics.py finished but checkpoint was not produced at $CKPT_PATH"
    echo "Check train_dynamics.py save path vs method.dynamics_ckpt in the yaml."
    exit 1
  fi
  echo "=> Dynamics ensemble ready: $CKPT_PATH"
else
  echo "=> Reusing existing dynamics ensemble at $CKPT_PATH (FORCE_RETRAIN_DYN=1 to rebuild)"
fi

# -------- Stage 2: Create / reuse wandb sweep and launch agents --------
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
  echo "Creating sweep from ${SWEEP_CONFIG} ..."
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
