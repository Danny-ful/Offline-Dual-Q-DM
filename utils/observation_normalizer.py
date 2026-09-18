"""Fixed observation normalization shared by offline data and environments.

The statistics intentionally match imitation-dice: they are fitted once from
the current states of the imperfect/supplement dataset and use
``(x - mean) / (std + eps)``. Offline replay buffers keep raw observations;
policy networks receive normalized views of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


class ObservationNormalizer:
    """Immutable per-feature affine observation transform."""

    def __init__(self, mean, std, eps: float = 1e-3, metadata: Optional[Mapping[str, Any]] = None):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        eps = float(eps)
        if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
            raise ValueError("Observation mean/std must be nonempty one-dimensional arrays")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Observation normalization statistics must be finite")
        if (std < 0).any() or not np.isfinite(eps) or eps <= 0:
            raise ValueError("Observation std must be nonnegative and eps must be finite and positive")
        self.mean = mean.copy()
        self.std = std.copy()
        self.eps = eps
        self.scale = (1.0 / (self.std + self.eps)).astype(np.float32)
        self.metadata = dict(metadata or {})
        self._torch_cache = {}
        for value in (self.mean, self.std, self.scale):
            value.setflags(write=False)

    @classmethod
    def fit(cls, observations, eps: float = 1e-3, metadata: Optional[Mapping[str, Any]] = None):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] == 0:
            raise ValueError("Observation normalization requires a nonempty [N, obs_dim] array")
        if not np.isfinite(observations).all():
            raise ValueError("Observation normalization data contains NaN or infinity")
        # NumPy's population standard deviation (ddof=0) matches imitation-dice.
        return cls(observations.mean(axis=0), observations.std(axis=0), eps, metadata)

    @property
    def obs_dim(self) -> int:
        return int(self.mean.shape[0])

    def _check_last_dim(self, observations) -> None:
        if observations.ndim == 0 or observations.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Observation width mismatch: got {tuple(observations.shape)}, expected last dimension {self.obs_dim}"
            )

    def normalize_np(self, observations):
        observations = np.asarray(observations)
        self._check_last_dim(observations)
        result = (observations - self.mean) * self.scale
        if not np.isfinite(result).all():
            raise ValueError("Observation normalization produced NaN or infinity")
        return result.astype(np.float32, copy=False)

    def denormalize_np(self, observations):
        observations = np.asarray(observations)
        self._check_last_dim(observations)
        result = observations / self.scale + self.mean
        if not np.isfinite(result).all():
            raise ValueError("Observation denormalization produced NaN or infinity")
        return result.astype(np.float32, copy=False)

    def _torch_stats(self, observations: torch.Tensor):
        self._check_last_dim(observations)
        key = (observations.device, observations.dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = (
                torch.tensor(self.mean, device=observations.device, dtype=observations.dtype),
                torch.tensor(self.scale, device=observations.device, dtype=observations.dtype),
            )
        return self._torch_cache[key]

    def normalize_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        mean, scale = self._torch_stats(observations)
        return (observations - mean) * scale

    def denormalize_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        mean, scale = self._torch_stats(observations)
        return observations / scale + mean

    def state_dict(self) -> dict:
        return {
            "format_version": 1,
            "mean": self.mean.copy(),
            "std": self.std.copy(),
            "eps": self.eps,
            "metadata": dict(self.metadata),
        }

    def save(self, path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format_version=np.asarray(1, dtype=np.int64),
            mean=self.mean,
            std=self.std,
            eps=np.asarray(self.eps, dtype=np.float64),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return str(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload["format_version"].item())
            if version != 1:
                raise ValueError(f"Unsupported observation normalizer format version: {version}")
            metadata_value = payload["metadata"].item()
            metadata = json.loads(str(metadata_value)) if metadata_value else {}
            return cls(payload["mean"], payload["std"], float(payload["eps"].item()), metadata)

    def assert_compatible(self, other: "ObservationNormalizer", *, rtol=1e-6, atol=1e-7) -> None:
        if self.obs_dim != other.obs_dim or not np.isclose(self.eps, other.eps, rtol=0, atol=0):
            raise ValueError("Observation normalizer dimension or epsilon does not match the checkpoint")
        if not np.allclose(self.mean, other.mean, rtol=rtol, atol=atol):
            raise ValueError("Observation normalizer mean does not match the checkpoint")
        if not np.allclose(self.std, other.std, rtol=rtol, atol=atol):
            raise ValueError("Observation normalizer std does not match the checkpoint")

    def summary(self) -> dict:
        near_constant = int(np.count_nonzero(self.std < self.eps))
        return {
            "obs_dim": self.obs_dim,
            "eps": self.eps,
            "std_min": float(self.std.min()),
            "std_max": float(self.std.max()),
            "scale_max": float(self.scale.max()),
            "near_constant_dims": near_constant,
        }


class NormalizedReplayView:
    """Normalize observations returned by a raw replay buffer exactly once."""

    def __init__(self, replay, normalizer: ObservationNormalizer):
        if isinstance(replay, NormalizedReplayView):
            raise ValueError("Replay buffer is already observation-normalized")
        self.replay = replay
        self.normalizer = normalizer

    def size(self):
        return self.replay.size()

    def get_samples(self, batch_size, device):
        state, next_state, action, reward, done = self.replay.get_samples(batch_size, device)
        return (
            self.normalizer.normalize_tensor(state),
            self.normalizer.normalize_tensor(next_state),
            action,
            reward,
            done,
        )

    def raw_observations(self):
        return self.replay.raw_observations()

    def __getattr__(self, name):
        return getattr(self.replay, name)


def policy_to_raw_observation(agent, observations: torch.Tensor) -> torch.Tensor:
    normalizer = getattr(agent, "observation_normalizer", None)
    return observations if normalizer is None else normalizer.denormalize_tensor(observations)


def raw_to_policy_observation(agent, observations: torch.Tensor) -> torch.Tensor:
    normalizer = getattr(agent, "observation_normalizer", None)
    return observations if normalizer is None else normalizer.normalize_tensor(observations)


def normalizer_checkpoint_path(base_path: str, suffix: str = "") -> str:
    return f"{base_path}{suffix}_obs_normalizer.npz"


def save_agent_observation_normalizer(agent, base_path: str, suffix: str = "") -> Optional[str]:
    normalizer = getattr(agent, "observation_normalizer", None)
    if normalizer is None:
        return None
    return normalizer.save(normalizer_checkpoint_path(base_path, suffix))


def validate_agent_observation_normalizer(agent, base_path: str, suffix: str = "") -> None:
    normalizer = getattr(agent, "observation_normalizer", None)
    path = normalizer_checkpoint_path(base_path, suffix)
    if normalizer is None:
        if Path(path).is_file():
            raise ValueError(
                "Checkpoint requires normalized observations, but observation "
                f"normalization is disabled: {path}"
            )
        return
    if Path(path).is_file():
        normalizer.assert_compatible(ObservationNormalizer.load(path))
        return
    cfg = getattr(agent.args, "observation_normalization", None)
    if cfg is not None and bool(getattr(cfg, "strict_checkpoint", True)):
        raise FileNotFoundError(f"Observation normalizer checkpoint not found: {path}")


def build_observation_normalizer(config, observations=None, *, stats_path=None, metadata=None):
    """Load configured fixed statistics or fit them from raw supplement states."""
    if config is None or not bool(getattr(config, "enabled", False)):
        return None
    configured_path = stats_path or getattr(config, "stats_path", None)
    if configured_path:
        normalizer = ObservationNormalizer.load(configured_path)
    else:
        if observations is None:
            raise ValueError(
                "Observation normalization is enabled but no supplement observations or stats_path were provided"
            )
        normalizer = ObservationNormalizer.fit(
            observations, eps=float(getattr(config, "eps", 1e-3)), metadata=metadata)
    summary = normalizer.summary()
    print(
        "--> Observation normalization enabled: "
        f"dim={summary['obs_dim']}, eps={summary['eps']}, "
        f"std=[{summary['std_min']:.6g}, {summary['std_max']:.6g}], "
        f"max_scale={summary['scale_max']:.6g}, "
        f"near_constant_dims={summary['near_constant_dims']}"
    )
    return normalizer
