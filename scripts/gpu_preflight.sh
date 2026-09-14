#!/usr/bin/env bash

# Shared GPU readiness check for server entry points.
# This file is intended to be sourced after activating the Python environment.

gpu_preflight() {
  local python_bin="${1:-${PYTHON_BIN:-python}}"
  local wait_seconds="${GPU_WAIT_SECONDS:-${WAIT_FOR_MOUNT_SECONDS:-120}}"
  local retry_seconds="${GPU_RETRY_SECONDS:-5}"
  local elapsed=0
  local attempt=1
  local torch_output=""
  local failure_reason=""

  if ! [[ "$wait_seconds" =~ ^[0-9]+$ ]]; then
    echo "[GPU preflight] GPU_WAIT_SECONDS must be a non-negative integer; got: $wait_seconds" >&2
    return 1
  fi
  if ! [[ "$retry_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "[GPU preflight] GPU_RETRY_SECONDS must be a positive integer; got: $retry_seconds" >&2
    return 1
  fi
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[GPU preflight] Python executable not found: $python_bin" >&2
    return 1
  fi

  echo "[GPU preflight] Waiting up to ${wait_seconds}s for CUDA (retry every ${retry_seconds}s)."

  while :; do
    failure_reason=""

    case "${CUDA_VISIBLE_DEVICES-__unset__}" in
      ""|-1|none|None|void|Void)
        failure_reason="CUDA_VISIBLE_DEVICES disables all GPUs (value: '${CUDA_VISIBLE_DEVICES-}')."
        ;;
    esac

    if [[ -z "$failure_reason" ]] && [[ ! -e /dev/nvidiactl ]]; then
      failure_reason="/dev/nvidiactl is missing; the cloud instance/container has not exposed the NVIDIA device."
    fi
    if [[ -z "$failure_reason" ]] && ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
      failure_reason="No /dev/nvidiaN device exists; no physical GPU is mounted in this runtime."
    fi

    if [[ -z "$failure_reason" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      if ! nvidia-smi >/dev/null 2>&1; then
        failure_reason="nvidia-smi cannot communicate with the NVIDIA driver; the driver may still be starting or may have reset."
      fi
    elif [[ -z "$failure_reason" ]]; then
      failure_reason="nvidia-smi is not installed or is not in PATH."
    fi

    if [[ -z "$failure_reason" ]]; then
      if torch_output="$("$python_bin" - <<'PY' 2>&1
import sys

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(10)

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")

if torch.version.cuda is None:
    print("This is a CPU-only PyTorch build.")
    raise SystemExit(11)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() returned False.")
    raise SystemExit(12)
if torch.cuda.device_count() < 1:
    print("torch.cuda.device_count() returned zero.")
    raise SystemExit(13)

try:
    tensor = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"allocation_device={tensor.device}")
except Exception as exc:
    print(f"CUDA allocation failed: {type(exc).__name__}: {exc}")
    raise SystemExit(14)
PY
      )"; then
        echo "[GPU preflight] CUDA is ready on attempt ${attempt}:"
        printf '%s\n' "$torch_output" | sed 's/^/[GPU preflight]   /'
        return 0
      else
        failure_reason="PyTorch could not initialize and allocate on cuda:0."
      fi
    fi

    echo "[GPU preflight] Attempt ${attempt} failed after ${elapsed}s: ${failure_reason}" >&2
    if [[ -n "$torch_output" ]]; then
      printf '%s\n' "$torch_output" | sed 's/^/[GPU preflight]   /' >&2
    fi

    if (( elapsed >= wait_seconds )); then
      break
    fi
    local sleep_seconds="$retry_seconds"
    if (( elapsed + sleep_seconds > wait_seconds )); then
      sleep_seconds=$((wait_seconds - elapsed))
    fi
    if (( sleep_seconds <= 0 )); then
      break
    fi
    sleep "$sleep_seconds"
    elapsed=$((elapsed + sleep_seconds))
    attempt=$((attempt + 1))
    torch_output=""
  done

  echo "[GPU preflight] ERROR: GPU was not ready within ${wait_seconds}s; refusing to run on CPU." >&2
  echo "[GPU preflight] Final reason: ${failure_reason}" >&2
  echo "[GPU preflight] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}" >&2
  echo "[GPU preflight] Python: $(command -v "$python_bin")" >&2
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU preflight] nvidia-smi diagnostic:" >&2
    nvidia-smi >&2 || true
  else
    echo "[GPU preflight] nvidia-smi: command not found" >&2
  fi
  echo "[GPU preflight] Check the cloud GPU attachment/container runtime and run: dmesg -T | grep -iE 'NVRM|Xid|fallen off'" >&2
  return 1
}
