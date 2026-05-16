"""Offline training of an N-member probabilistic dynamics ensemble.

The ensemble is trained once on the full offline dataset (expert + supplement)
and saved to disk. During IQ training the checkpoint is simply loaded and
frozen (see `train_iq.py`).

Usage:
    python train_dynamics.py env=hopper method=iq method.penalty_N=5 \
        dyn.epochs=100 dyn.lr=1e-3 dyn.batch_size=256

Notes:
    * Reuses the same expert / supplement loading logic as ``train_iq.py`` so
      the produced checkpoint is perfectly aligned with the IQ training data.
    * Each ensemble member is trained with an independent bootstrap resample
      of the full dataset.
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from agent.dynamics_ensemble import DynamicsEnsemble
from dataset.memory import Memory
from make_envs import make_env


def _collect_transitions(memory: Memory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a Memory buffer into (obs, action, next_obs) arrays."""
    obs_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    for item in memory.buffer:
        state, next_state, action, _reward, _done = item
        obs_list.append(np.asarray(state, dtype=np.float32))
        next_obs_list.append(np.asarray(next_state, dtype=np.float32))
        action_list.append(np.asarray(action, dtype=np.float32))
    obs = np.stack(obs_list, axis=0)
    next_obs = np.stack(next_obs_list, axis=0)
    actions = np.stack(action_list, axis=0)
    if actions.ndim == 1:
        actions = actions[:, None]
    return obs, actions, next_obs


def _build_dataset(cfg: DictConfig, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    REPLAY_MEMORY = int(cfg.env.replay_mem)
    demo_filename = os.path.basename(cfg.env.demo)

    expert_memory = Memory(REPLAY_MEMORY // 2, seed)
    expert_memory.load(
        hydra.utils.to_absolute_path(f"experts/{demo_filename}"),
        num_trajs=cfg.expert.demos,
        sample_freq=cfg.expert.subsample_freq,
        seed=seed + 42,
    )
    print(f"--> Expert memory size: {expert_memory.size()}")

    supplement_path = hydra.utils.to_absolute_path(f"supplement/{demo_filename}")
    if not os.path.isfile(supplement_path):
        raise FileNotFoundError(
            f"Supplement dataset not found at {supplement_path}. "
            "Train_dynamics expects the same offline data as train_iq in offline mode."
        )
    supplement_memory = Memory(REPLAY_MEMORY // 2, seed + 1)
    supplement_memory.load(
        supplement_path,
        num_trajs=np.iinfo(np.int32).max,
        sample_freq=cfg.expert.subsample_freq,
        seed=seed + 43,
    )
    print(f"--> Supplement memory size: {supplement_memory.size()}")

    e_obs, e_act, e_next = _collect_transitions(expert_memory)
    s_obs, s_act, s_next = _collect_transitions(supplement_memory)

    obs = np.concatenate([e_obs, s_obs], axis=0)
    actions = np.concatenate([e_act, s_act], axis=0)
    next_obs = np.concatenate([e_next, s_next], axis=0)
    print(f"--> Total transitions for dynamics training: {obs.shape[0]}")
    return obs, actions, next_obs


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # We only need env to query observation/action dims.
    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dyn_cfg = cfg.get("dyn", {}) or {}
    N = int(getattr(cfg.method, "penalty_N", dyn_cfg.get("N", 5)))
    epochs = int(dyn_cfg.get("epochs", 100))
    batch_size = int(dyn_cfg.get("batch_size", 256))
    lr = float(dyn_cfg.get("lr", 1e-3))
    weight_decay = float(dyn_cfg.get("weight_decay", 1e-5))
    hidden_dim = int(dyn_cfg.get("hidden_dim", 256))
    hidden_depth = int(dyn_cfg.get("hidden_depth", 3))
    val_frac = float(dyn_cfg.get("val_frac", 0.05))
    log_interval = int(dyn_cfg.get("log_interval", 5))

    obs, actions, next_obs = _build_dataset(cfg, seed=cfg.seed)
    num_samples = obs.shape[0]
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=cfg.device)
    next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=cfg.device)

    # Hold-out validation split (same across members; per-member bootstrap on train).
    perm = torch.randperm(num_samples, device=cfg.device)
    num_val = max(1, int(num_samples * val_frac)) if val_frac > 0 else 0
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    val_obs = obs_t[val_idx]
    val_act = act_t[val_idx]
    val_next = next_t[val_idx]

    ensemble = DynamicsEnsemble(
        obs_dim=obs_dim,
        action_dim=action_dim,
        N=N,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
    ).to(cfg.device)

    # Optimize each member independently.
    optimizers = [
        torch.optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay)
        for member in ensemble.members
    ]

    num_train = train_idx.numel()
    print(f"--> Dataset: train={num_train} val={num_val} | N={N} epochs={epochs} bs={batch_size}")

    start_time = time.time()
    for epoch in range(epochs):
        # Per-member bootstrap: independent random index ordering each epoch.
        member_losses: List[float] = []
        for i, (member, opt) in enumerate(zip(ensemble.members, optimizers)):
            member.train()
            # Bootstrap with replacement across the train split.
            idx = train_idx[torch.randint(0, num_train, (num_train,), device=cfg.device)]
            total, count = 0.0, 0
            for start in range(0, num_train, batch_size):
                batch_idx = idx[start : start + batch_size]
                b_obs = obs_t[batch_idx]
                b_act = act_t[batch_idx]
                b_next = next_t[batch_idx]
                loss = member.nll_loss(b_obs, b_act, b_next)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * b_obs.size(0)
                count += b_obs.size(0)
            member_losses.append(total / max(count, 1))

        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            val_line = ""
            if num_val > 0:
                with torch.no_grad():
                    val_losses = []
                    for member in ensemble.members:
                        member.eval()
                        val_losses.append(float(member.nll_loss(val_obs, val_act, val_next).item()))
                val_line = (
                    f" | val mean={np.mean(val_losses):.4f} "
                    f"min={np.min(val_losses):.4f} max={np.max(val_losses):.4f}"
                )
            elapsed = time.time() - start_time
            print(
                f"[dynamics] epoch {epoch + 1:>4d}/{epochs} "
                f"train mean={np.mean(member_losses):.4f} "
                f"min={np.min(member_losses):.4f} max={np.max(member_losses):.4f}"
                f"{val_line} | elapsed {elapsed:.1f}s"
            )

    # Save checkpoint
    save_dir = hydra.utils.to_absolute_path(f"dynamics/{cfg.env.name}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ensemble_{N}.pt")
    ensemble.save(save_path)
    print(f"--> Saved ensemble checkpoint to {save_path}")


if __name__ == "__main__":
    main()
