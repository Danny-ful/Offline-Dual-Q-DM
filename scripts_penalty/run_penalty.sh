#!/usr/bin/env bash
set -euo pipefail

# One-click single-run with the dynamics uncertainty penalty enabled.
#
# Workflow:
#   1) If the dynamics ensemble checkpoint is missing (or FORCE_RETRAIN_DYN=1),
#      pretrain it with train_dynamics.py on expert + supplement data.
#   2) Run train_iq.py with method.penalty=True using that checkpoint.
#
# Environment knobs (all optional):
#   DYN_CKPT=dynamics/Hopper-v2/ensemble_5.pt   # where to save / look for ensemble
#   FORCE_RETRAIN_DYN=0                      # set to 1 to force re-training
#   PENALTY_N=5                              # ensemble size (must match ckpt)
#   PENALTY_M=5                              # samples per member at IQ time
#   PENALTY_COEF=0.01                        # uncertainty-penalty scale
#   EXPERT_DEMOS_DYN=1                       # expert.demos for dynamics pretraining
#   EXPERT_DEMOS_IQ=1                        # expert.demos for IQ stage
#   DYN_EPOCHS=200                           # epochs for dynamics pretraining
#   SEED=0                                   # run seed
#   LEARN_STEPS=800000
#   EVAL_INTERVAL=5000
#   EVAL_EPS=20
#   EXP_NAME=hopper_penalty_1p5k_v1

WAIT_FOR_MOUNT_SECONDS="${WAIT_FOR_MOUNT_SECONDS:-15}"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No python executable found in PATH"
  exit 1
fi

PENALTY_N="${PENALTY_N:-5}"
PENALTY_M="${PENALTY_M:-10}"
PENALTY_COEF="${PENALTY_COEF:-3}"
EXPERT_DEMOS_DYN="${EXPERT_DEMOS_DYN:-1}"
EXPERT_DEMOS_IQ="${EXPERT_DEMOS_IQ:-1}"
DYN_EPOCHS="${DYN_EPOCHS:-200}"
DYN_CKPT="${DYN_CKPT:-dynamics/Hopper-v2/ensemble_${PENALTY_N}.pt}"
FORCE_RETRAIN_DYN="${FORCE_RETRAIN_DYN:-0}"
SEED="${SEED:-0}"
LEARN_STEPS="${LEARN_STEPS:-500000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_EPS="${EVAL_EPS:-20}"
EXP_NAME="${EXP_NAME:-hopper_penalty_1p5k_v1}"

# -------- Stage 1: Dynamics ensemble pretraining (auto) --------
if [ ! -f "$DYN_CKPT" ] || [ "$FORCE_RETRAIN_DYN" = "1" ]; then
  if [ "$FORCE_RETRAIN_DYN" = "1" ]; then
    echo "=> FORCE_RETRAIN_DYN=1, (re)training dynamics ensemble..."
  else
    echo "=> Dynamics ensemble not found at $DYN_CKPT, training..."
  fi
  "$PYTHON_BIN" train_dynamics.py env=hopper agent=sac offline=True \
    env.demo=Hopper-v2_d4rl.pkl expert.demos="$EXPERT_DEMOS_DYN" expert.subsample_freq=1 \
    method=iq method.penalty_N="$PENALTY_N" \
    +dyn.epochs="$DYN_EPOCHS" +dyn.batch_size=512 +dyn.lr=5e-4 \
    seed="$SEED"
  if [ ! -f "$DYN_CKPT" ]; then
    echo "train_dynamics.py finished but checkpoint was not produced at $DYN_CKPT"
    echo "Check train_dynamics.py output and save path consistency."
    exit 1
  fi
  echo "=> Dynamics ensemble ready: $DYN_CKPT"
else
  echo "=> Reusing existing dynamics ensemble at $DYN_CKPT"
fi

# -------- Stage 2: IQ training with penalty enabled --------
"$PYTHON_BIN" train_iq.py env=hopper agent=sac offline=True \
  env.demo=Hopper-v2_d4rl.pkl expert.demos="$EXPERT_DEMOS_IQ" expert.subsample_freq=1 \
  method.loss=value method.constrain=True \
  method.grad_pen=True method.lambda_gp=10 \
  train.batch=512 train.use_target=True train.soft_update=True \
  agent.critic_lr=5e-4 agent.critic_tau=0.01 agent.critic_target_update_frequency=1 \
  agent.actor_lr=3e-5 agent.actor_update_frequency=2 num_actor_updates=1 \
  agent.init_temp=0.1 \
  cliptarget=True gamma=0.99 \
  penalty=5 left=-0.9 right=0.9 \
  actor_expert_offline=False \
  schedular=False cuda_deterministic=True \
  env.learn_steps="$LEARN_STEPS" env.eval_interval="$EVAL_INTERVAL" eval.eps="$EVAL_EPS" \
  method.penalty=True \
  method.penalty_N="$PENALTY_N" method.penalty_M="$PENALTY_M" method.penalty_coef="$PENALTY_COEF" \
  method.dynamics_ckpt="$DYN_CKPT" \
  project_name=Offline-Dual-Q-DM exp_name="$EXP_NAME" seed="$SEED"
