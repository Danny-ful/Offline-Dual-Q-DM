"""Probabilistic dynamics ensemble used to build an uncertainty penalty U(s, a).

Each member is a Gaussian MLP that predicts the delta over obs, i.e.
    s'_hat = s + mean(s, a) + exp(log_std(s, a)) * epsilon,
with learnable per-dimension log_std bounds (MOPO-style).
"""

from __future__ import annotations

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
        # Identity defaults preserve the behavior of unnormalized/legacy models.
        for name, dim in (("obs", obs_dim), ("action", action_dim), ("delta", obs_dim)):
            self.register_buffer(name + "_mean", torch.zeros(dim))
            self.register_buffer(name + "_std", torch.ones(dim))

    @torch.no_grad()
    def fit_normalization(self, obs, action, next_obs, eps: float = 1e-6) -> None:
        """Fit once using the training split, before member bootstrapping."""
        if eps <= 0 or not torch.isfinite(torch.tensor(eps)):
            raise ValueError("Normalization eps must be finite and positive")
        for name, values in (("obs", obs), ("action", action), ("delta", next_obs - obs)):
            if len(values) == 0 or not torch.isfinite(values).all():
                raise ValueError("Normalization requires nonempty, finite training data")
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False)
            # Constant features should not amplify tiny numerical perturbations.
            std = torch.where(std < eps, torch.ones_like(std), std)
            getattr(self, name + "_mean").copy_(mean)
            getattr(self, name + "_std").copy_(std)

    def _forward_raw(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([(obs - self.obs_mean) / self.obs_std,
                       (action - self.action_mean) / self.action_std], dim=-1)
        mean, log_std = self.trunk(x).chunk(2, dim=-1)
        # soft-bound log_std to [min_log_std, max_log_std]
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return mean, log_std

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Return (mean_delta, log_std_delta) in original observation units."""
        mean, log_std = self._forward_raw(obs, action)
        return mean * self.delta_std + self.delta_mean, log_std + self.delta_std.log()

    def nll_per_sample(self, obs, action, next_obs):
        """Normalized Gaussian NLL (twice NLL, omitting the constant), no regularizer."""
        mean, log_std = self._forward_raw(obs, action)
        target_delta = (next_obs - obs - self.delta_mean) / self.delta_std
        return (((mean - target_delta) ** 2) * torch.exp(-2.0 * log_std)
                + 2.0 * log_std).sum(dim=-1)

    def nll_loss(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        loss = self.nll_per_sample(obs, action, next_obs).mean()
        # small regularizer to keep the learned bounds from drifting apart too much
        reg = 0.01 * (self.max_log_std.sum() - self.min_log_std.sum())
        return loss + reg

    @torch.no_grad()
    def sample_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs, action)
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
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
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

    def save(self, path: str, training_metadata: Optional[dict] = None) -> None:
        payload = {
            "format_version": 2,
            "state_dict": self.state_dict(),
            "cfg": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "N": self.N,
                "effective_obs_dim": self.effective_obs_dim,
                "hidden_dim": self.hidden_dim,
                "hidden_depth": self.hidden_depth,
            },
            "training_metadata": training_metadata or {},
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
            state_dict = payload["state_dict"]
        else:
            # backward compat: raw state dict
            state_dict = payload
        normalization_keys = [f"members.{i}.{name}_{stat}" for i in range(self.N)
                              for name in ("obs", "action", "delta")
                              for stat in ("mean", "std")]
        # Only genuinely old checkpoints may omit all normalization buffers.
        # A partially missing set in a new checkpoint is an error, not an identity fallback.
        if not any(key in state_dict for key in normalization_keys) and (
                not isinstance(payload, dict) or payload.get("format_version", 1) < 2):
            state_dict = dict(state_dict)
            defaults = self.state_dict()
            for key in normalization_keys:
                state_dict[key] = (torch.ones_like(defaults[key]) if key.endswith("_std")
                                   else torch.zeros_like(defaults[key]))
        self.load_state_dict(state_dict)


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
    dataset = payload.get('training_metadata', {}).get('dataset', {}) if isinstance(payload, dict) else {}
    robosuite = getattr(args, 'robosuite', None)
    if robosuite is not None and dataset:
        if dataset.get('task') != str(robosuite.task).lower():
            raise ValueError('Dynamics checkpoint task does not match robosuite.task')
        if dataset.get('obs_keys') != list(robosuite.obs_keys):
            raise ValueError('Dynamics checkpoint observation order does not match robosuite.obs_keys')
        if payload.get('cfg', {}).get('obs_dim') != obs_dim:
            raise ValueError('Dynamics checkpoint observation dimension does not match Robosuite IQ')
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
