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
        effective_obs_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.N = N
        # When set, the model only operates on the first effective_obs_dim dims
        # and pads the rest with zeros on output.
        self.effective_obs_dim = effective_obs_dim or obs_dim
        self.members = nn.ModuleList(
            [
                ProbDynamics(
                    self.effective_obs_dim,
                    action_dim,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                )
                for _ in range(N)
            ]
        )

    def _slice_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.effective_obs_dim < self.obs_dim:
            return obs[..., : self.effective_obs_dim]
        return obs

    def _pad_obs(self, obs_short: torch.Tensor, full_obs: torch.Tensor) -> torch.Tensor:
        if self.effective_obs_dim < self.obs_dim:
            pad = full_obs[..., self.effective_obs_dim:]
            return torch.cat([obs_short, pad], dim=-1)
        return obs_short

    def forward(self, i: int, obs: torch.Tensor, action: torch.Tensor):
        return self.members[i](self._slice_obs(obs), action)

    @torch.no_grad()
    def sample_next_ensemble(
        self, obs: torch.Tensor, action: torch.Tensor, M: int = 1,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [B, N, M, obs_dim]; optional [B, M, effective_obs_dim]
        standard normal noise is shared across members. Otherwise sample independently.
        """
        if M < 1:
            raise ValueError("M must be at least 1")
        B = obs.size(0)
        if noise is not None:
            if noise.shape != (B, M, self.effective_obs_dim):
                raise ValueError("Invalid dynamics noise shape")
            noise = noise.to(obs).reshape(B * M, self.effective_obs_dim)
        obs_eff = self._slice_obs(obs)
        obs_rep = obs_eff.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
        action_rep = action.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)

        out = obs.new_empty(B, self.N, M, self.obs_dim)
        # Keep the ignored dims from the original obs for padding
        pad_dims = obs[:, self.effective_obs_dim:]  # [B, obs_dim - eff]
        for i, member in enumerate(self.members):
            mean, log_std = member(obs_rep, action_rep)
            eps = torch.randn_like(mean) if noise is None else noise
            s_next_short = obs_rep + mean + torch.exp(log_std) * eps  # [B*M, eff]
            s_next_short = s_next_short.view(B, M, self.effective_obs_dim)
            if self.effective_obs_dim < self.obs_dim:
                pad = pad_dims.unsqueeze(1).expand(B, M, -1)
                s_next_full = torch.cat([s_next_short, pad], dim=-1)
            else:
                s_next_full = s_next_short
            out[:, i] = s_next_full
        return out

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "cfg": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "N": self.N,
                "effective_obs_dim": self.effective_obs_dim,
            },
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            cfg = payload.get("cfg", {})
            if cfg:
                assert cfg.get("action_dim") == self.action_dim, (
                    f"action_dim mismatch: ckpt={cfg.get('action_dim')} vs module={self.action_dim}"
                )
                assert cfg.get("N") == self.N, (
                    f"ensemble size mismatch: ckpt N={cfg.get('N')} vs module N={self.N}"
                )
                ckpt_eff = cfg.get("effective_obs_dim", cfg.get("obs_dim"))
                assert ckpt_eff == self.effective_obs_dim, (
                    f"effective_obs_dim mismatch: ckpt={ckpt_eff} vs module={self.effective_obs_dim}"
                )
            self.load_state_dict(payload["state_dict"])
        else:
            # backward compat: raw state dict
            self.load_state_dict(payload)


def load_iq_dynamics(agent, obs_dim, action_dim):
    """Load the frozen model when either IQ dynamics-based loss is enabled."""
    import hydra
    from iq import validate_synthetic_config

    args = agent.args
    method = args.method
    validate_synthetic_config(agent)
    if not (getattr(method, 'uncertainty', False)
            or getattr(method, 'synthetic_constrain', False)):
        return
    if not method.dynamics_ckpt:
        raise ValueError('Dynamics-based IQ losses require method.dynamics_ckpt; run train_dynamics.py first')
    path = hydra.utils.to_absolute_path(method.dynamics_ckpt)
    payload = torch.load(path, map_location='cpu')
    cfg = payload.get('cfg', {}) if isinstance(payload, dict) else {}
    state_dict = payload.get('state_dict', payload)
    hidden_dim = cfg.get('hidden_dim')
    if hidden_dim is None:
        hidden_dim = state_dict['members.0.trunk.0.weight'].shape[0]
    hidden_depth = cfg.get('hidden_depth')
    if hidden_depth is None:
        hidden_depth = sum(key.startswith('members.0.trunk.') and key.endswith('.weight')
                           for key in state_dict) - 1
    ensemble = DynamicsEnsemble(
        obs_dim, action_dim, N=int(method.penalty_N),
        effective_obs_dim=cfg.get('effective_obs_dim'),
        hidden_dim=hidden_dim, hidden_depth=hidden_depth,
    ).to(args.device)
    ensemble.load(path, map_location=args.device)
    ensemble.eval()
    ensemble.requires_grad_(False)
    agent.dynamics_ensemble = ensemble
    print(f'--> Loaded dynamics ensemble (N={method.penalty_N}) from {path}')
