"""Environment wrapper for a fixed offline-data observation transform."""

from __future__ import annotations

import numpy as np


class NormalizeObservationWrapper:
    """Normalize reset/step observations while preserving rewards and episode info."""

    def __init__(self, env, normalizer):
        if isinstance(env, NormalizeObservationWrapper):
            raise ValueError("Environment is already observation-normalized")
        self.env = env
        self.normalizer = normalizer
        observation_space = getattr(env, "observation_space", None)
        if observation_space is not None:
            if observation_space.shape != (normalizer.obs_dim,):
                raise ValueError(
                    f"Environment observation shape {observation_space.shape} does not match "
                    f"normalizer dimension {normalizer.obs_dim}"
                )
            try:
                import gym
                low = normalizer.normalize_np(observation_space.low)
                high = normalizer.normalize_np(observation_space.high)
                self.observation_space = gym.spaces.Box(
                    low=np.minimum(low, high),
                    high=np.maximum(low, high),
                    dtype=np.float32,
                )
            except (ValueError, TypeError, OverflowError):
                import gym
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(normalizer.obs_dim,), dtype=np.float32
                )

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            observation, info = result
            return self.normalizer.normalize_np(observation), info
        return self.normalizer.normalize_np(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return (self.normalizer.normalize_np(observation), reward, terminated, truncated, info)
        observation, reward, done, info = result
        return self.normalizer.normalize_np(observation), reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)
