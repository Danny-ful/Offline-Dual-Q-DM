"""Probabilistic dynamics ensemble used to build an uncertainty penalty U(s, a).

Each member is a Gaussian MLP that predicts the delta over obs, i.e.
    s'_hat = s + mean(s, a) + exp(log_std(s, a)) * epsilon,
with learnable per-dimension log_std bounds (MOPO-style).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_depth: int) -> nn.Sequential:
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    mods: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


class ProbDynamics(nn.Module):
    """Gaussian dynamics head predicting delta-state."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.trunk = _mlp(obs_dim + action_dim, hidden_dim, 2 * obs_dim, hidden_depth)

        # learnable per-dim log_std bounds (MOPO trick keeps log_std within a range)
        self.max_log_std = nn.Parameter(torch.full((obs_dim,), log_std_max))
        self.min_log_std = nn.Parameter(torch.full((obs_dim,), log_std_min))

    def _forward_raw(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        mean, log_std = self.trunk(x).chunk(2, dim=-1)
        # soft-bound log_std to [min_log_std, max_log_std]
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return mean, log_std

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Return (mean_delta, log_std_delta)."""
        return self._forward_raw(obs, action)

    def nll_loss(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        mean, log_std = self._forward_raw(obs, action)
        target_delta = next_obs - obs
        inv_var = torch.exp(-2.0 * log_std)
        # Gaussian NLL (up to constants). Sum over obs dim, mean over batch.
        nll = ((mean - target_delta) ** 2) * inv_var + 2.0 * log_std
        loss = nll.sum(dim=-1).mean()
        # small regularizer to keep the learned bounds from drifting apart too much
        reg = 0.01 * (self.max_log_std.sum() - self.min_log_std.sum())
        return loss + reg

    @torch.no_grad()
    def sample_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self._forward_raw(obs, action)
        eps = torch.randn_like(mean)
        delta = mean + torch.exp(log_std) * eps
        return obs + delta


class DynamicsEnsemble(nn.Module):
    """Ensemble of N ProbDynamics heads trained independently."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        N: int = 5,
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.N = N
        self.members = nn.ModuleList(
            [
                ProbDynamics(
                    obs_dim,
                    action_dim,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                )
                for _ in range(N)
            ]
        )

    def forward(self, i: int, obs: torch.Tensor, action: torch.Tensor):
        return self.members[i](obs, action)

    @torch.no_grad()
    def sample_next_ensemble(
        self, obs: torch.Tensor, action: torch.Tensor, M: int = 1
    ) -> torch.Tensor:
        """Return samples of shape [B, N, M, obs_dim]."""
        B = obs.size(0)
        # broadcast obs/action over M samples by feeding the same (obs,action) M times
        obs_rep = obs.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
        action_rep = action.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)

        out = obs.new_empty(B, self.N, M, self.obs_dim)
        for i, member in enumerate(self.members):
            mean, log_std = member(obs_rep, action_rep)
            eps = torch.randn_like(mean)
            s_next = obs_rep + mean + torch.exp(log_std) * eps
            out[:, i] = s_next.view(B, M, self.obs_dim)
        return out

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "cfg": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "N": self.N,
            },
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            cfg = payload.get("cfg", {})
            if cfg:
                assert cfg.get("obs_dim") == self.obs_dim, (
                    f"obs_dim mismatch: ckpt={cfg.get('obs_dim')} vs module={self.obs_dim}"
                )
                assert cfg.get("action_dim") == self.action_dim, (
                    f"action_dim mismatch: ckpt={cfg.get('action_dim')} vs module={self.action_dim}"
                )
                assert cfg.get("N") == self.N, (
                    f"ensemble size mismatch: ckpt N={cfg.get('N')} vs module N={self.N}"
                )
            self.load_state_dict(payload["state_dict"])
        else:
            # backward compat: raw state dict
            self.load_state_dict(payload)
