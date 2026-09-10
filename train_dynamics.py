"""Train normalized dynamics with fixed member bootstraps and trajectory holdout.

Uses the same ExpertDataset selection/subsampling as IQ, retaining trajectory IDs.
Public model predictions and samples remain in original observation units.
"""
from __future__ import annotations

import json
import os
import random
import time
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dataset.expert_dataset import ExpertDataset


def _build_dataset(cfg: DictConfig, seed: int, obs_dim=None, action_dim=None,
                   original_obs_dim=None):
    """Align observations before stacking; keep source-qualified trajectory IDs.

    Accept raw or already-reduced observations when ReduceObsWrapper is active.
    Terminal flags may omit timeouts.
    """
    if int(cfg.expert.demos) < 1 or int(cfg.expert.subsample_freq) < 1:
        raise ValueError("expert.demos and expert.subsample_freq must be positive")
    arrays = [[], [], []]
    trajectory_ids, sources = [], []
    for source, path, demos, offset in (
        ("expert", cfg.env.expert_path, cfg.expert.demos, 42),
        ("supplement", cfg.env.supplement_path, np.iinfo(np.int32).max, 43),
    ):
        path = hydra.utils.to_absolute_path(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{source} dataset not found at {path}")
        data = ExpertDataset(path, demos, cfg.expert.subsample_freq, seed + offset)
        if not len(data):
            raise ValueError(f"{source} dataset has no transitions after subsampling")
        for index in range(len(data)):
            obs, next_obs, action, _, _ = data[index]
            obs, next_obs, action = (np.asarray(value, dtype=np.float32)
                                     for value in (obs, next_obs, action))
            if action.ndim == 0:
                action = action.reshape(1)
            if obs_dim is not None:
                allowed_obs_dims = {obs_dim, original_obs_dim or obs_dim}
                if (obs.ndim != 1 or obs.shape[0] not in allowed_obs_dims
                        or next_obs.shape != obs.shape
                        or action.shape != (action_dim,)):
                    raise ValueError(
                        f"{source} dataset at {path}, transition {index}: "
                        f"observation/action dimensions do not match the environment; "
                        f"got obs={obs.shape}, next_obs={next_obs.shape}, action={action.shape}; "
                        f"expected obs and next_obs dimensions {sorted(allowed_obs_dims)}, "
                        f"action=({action_dim},)"
                    )
                # Apply the same prefix projection as ReduceObsWrapper before
                # stacking, including when expert and supplement widths differ.
                obs, next_obs = obs[:obs_dim], next_obs[:obs_dim]
            for target, value in zip(arrays, (obs, action, next_obs)):
                target.append(value)
            trajectory_ids.append(f"{source}:{data.get_idx[index][0]}")
            sources.append(source)
        print(f"--> {source}: {len(data)} transitions")
    obs, actions, next_obs = (np.stack(values) for values in arrays)
    if actions.ndim == 1:
        actions = actions[:, None]
    if obs.ndim != 2 or next_obs.shape != obs.shape or actions.ndim != 2:
        raise ValueError("Dynamics requires flat observations and actions with matching next observations")
    if not all(np.isfinite(values).all() for values in (obs, actions, next_obs)):
        raise ValueError("Dynamics data contains NaN or infinity")
    return obs, actions, next_obs, np.asarray(trajectory_ids), np.asarray(sources)


def _split_trajectories(trajectory_ids, sources, val_frac, seed):
    """Hold out whole trajectories per source, retaining singletons for training.

    val_frac is a fraction of trajectories, so transition fractions may differ.
    """
    if not 0 <= val_frac < 1:
        raise ValueError("dyn.val_frac must be in [0, 1)")
    if len(trajectory_ids) == 0 or len(trajectory_ids) != len(sources):
        raise ValueError("Need nonempty, aligned trajectory IDs and sources")
    rng = np.random.default_rng(seed)
    held_out = []
    if val_frac > 0:
        for source in np.unique(sources):
            ids = np.unique(trajectory_ids[sources == source])
            if len(ids) < 2:
                warnings.warn(f"{source} has only one trajectory; retained in training, no holdout for this source")
                continue
            count = min(len(ids) - 1, max(1, int(len(ids) * val_frac)))
            held_out.extend(rng.permutation(ids)[:count].tolist())
        if not held_out:
            raise ValueError("Trajectory validation needs at least two trajectories in one source; "
                             "load more trajectories or explicitly set dyn.val_frac=0")
    mask = np.isin(trajectory_ids, held_out)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def _fixed_bootstrap(train_idx, members, seed):
    """Draw once. Epochs may permute these indices but never redraw them."""
    generator = torch.Generator(device=train_idx.device).manual_seed(seed)
    return [train_idx[torch.randint(len(train_idx), (len(train_idx),),
                                   generator=generator, device=train_idx.device)]
            for _ in range(members)]


@torch.no_grad()
def _validation_losses(ensemble, obs, actions, next_obs, indices, batch_size):
    totals = np.zeros(ensemble.N, dtype=np.float64)
    ensemble.eval()
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        for i, member in enumerate(ensemble.members):
            totals[i] += member.nll_per_sample(obs[idx], actions[idx], next_obs[idx]).double().sum().item()
    losses = totals / len(indices)
    if not np.isfinite(losses).all():
        raise FloatingPointError("Nonfinite validation NLL")
    return losses


def _train_ensemble(ensemble, obs, actions, next_obs, train_idx, val_idx, dyn_cfg, seed):
    epochs = int(dyn_cfg.get("epochs", 100))
    batch_size = int(dyn_cfg.get("batch_size", 256))
    log_interval = int(dyn_cfg.get("log_interval", 5))
    if min(epochs, batch_size, log_interval) < 1 or len(train_idx) == 0:
        raise ValueError("epochs, batch_size, log_interval and training size must be positive")
    eps = float(dyn_cfg.get("normalization_eps", 1e-6))
    # Shared training statistics, independent resamples and optimization per member.
    ensemble.members[0].fit_normalization(obs[train_idx], actions[train_idx], next_obs[train_idx], eps)
    for member in ensemble.members[1:]:
        for name, value in ensemble.members[0].named_buffers():
            getattr(member, name).copy_(value)
    optimizers = [torch.optim.Adam(member.parameters(), lr=float(dyn_cfg.get("lr", 1e-3)),
                                  weight_decay=float(dyn_cfg.get("weight_decay", 1e-5)))
                  for member in ensemble.members]
    bootstrap_seed = int(seed) + 1000
    bootstraps = _fixed_bootstrap(train_idx, ensemble.N, bootstrap_seed)
    shuffle_rng = torch.Generator(device=train_idx.device).manual_seed(int(seed) + 1001)
    best_losses = np.full(ensemble.N, np.inf)
    best_epochs = [None] * ensemble.N
    best_states = [None] * ensemble.N
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        member_losses = []
        for member, opt, bootstrap in zip(ensemble.members, optimizers, bootstraps):
            member.train()
            idx = bootstrap[torch.randperm(len(bootstrap), device=train_idx.device, generator=shuffle_rng)]
            total = 0.0
            for start in range(0, len(idx), batch_size):
                batch = idx[start:start + batch_size]
                loss = member.nll_loss(obs[batch], actions[batch], next_obs[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            member_losses.append(total / len(idx))
        val_losses = None
        if len(val_idx):
            # Evaluate every epoch regardless of console log frequency; exclude bound regularizer.
            val_losses = _validation_losses(ensemble, obs, actions, next_obs, val_idx, batch_size)
            for i, loss in enumerate(val_losses):
                if loss < best_losses[i]:
                    best_losses[i] = loss
                    best_epochs[i] = epoch
                    best_states[i] = {key: value.detach().cpu().clone()
                                      for key, value in ensemble.members[i].state_dict().items()}
        history.append({"epoch": epoch, "train_loss": member_losses,
                        "val_nll": None if val_losses is None else val_losses.tolist()})
        if epoch % log_interval == 0 or epoch == epochs:
            val_line = "" if val_losses is None else f" | val NLL={val_losses.round(4).tolist()}"
            print(f"[dynamics] epoch {epoch}/{epochs} train mean={np.mean(member_losses):.4f}"
                  f"{val_line} | elapsed {time.time() - start_time:.1f}s")
    if len(val_idx):
        for member, state in zip(ensemble.members, best_states):
            member.load_state_dict(state)
        print(f"--> Restored member best epochs: {best_epochs}")
    else:
        warnings.warn("Validation disabled: saving last-epoch weights; no best-weight selection or holdout diagnostics")
    ensemble.eval()
    return {"seed": int(seed), "bootstrap_seed": bootstrap_seed,
            "bootstrap_unique_counts": [int(torch.unique(idx).numel()) for idx in bootstraps],
            "selection_metric": "normalized_nll_without_regularizer" if len(val_idx) else None,
            "best_epochs": best_epochs,
            "best_val_nll": best_losses.tolist() if len(val_idx) else None,
            "history": history}


def _correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))


def _ranks(values):
    # Average ranks for ties (including constant disagreement from a single member).
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2.0)[inverse]


def _metric_summary(metrics, mask, bins):
    result = {"transitions": int(mask.sum())}
    for name, values in metrics.items():
        if name == "member_nll":
            result[name] = values[mask].mean(axis=0).tolist()
        else:
            result[name] = float(values[mask].mean())
    error, disagreement = metrics["normalized_mse"][mask], metrics["normalized_disagreement"][mask]
    result["error_disagreement_pearson"] = _correlation(error, disagreement)
    result["error_disagreement_spearman"] = _correlation(_ranks(error), _ranks(disagreement))
    # Quantile edges keep tied disagreements together instead of inventing an ordering.
    edges = np.unique(np.quantile(disagreement, np.linspace(0, 1, bins + 1)))
    groups = np.searchsorted(edges[1:-1], disagreement, side="right")
    result["disagreement_bins"] = []
    for group in np.unique(groups):
        selected = groups == group
        result["disagreement_bins"].append({
            "count": int(selected.sum()), "min": float(disagreement[selected].min()),
            "max": float(disagreement[selected].max()),
            "mean_disagreement": float(disagreement[selected].mean()),
            "mean_error": float(error[selected].mean()),
        })
    return result


@torch.no_grad()
def _diagnostics(ensemble, obs, actions, next_obs, val_idx, trajectory_ids, sources, batch_size, bins=10):
    """One-step holdout error vs variance of deterministic member means.

    Aleatoric variance is separate; random samples never enter disagreement.
    Normalized metrics use the training delta scale so features are comparable.
    """
    if batch_size < 1 or bins < 1:
        raise ValueError("Diagnostic batch size and bins must be positive")
    if not len(val_idx):
        return {"status": "disabled", "reason": "dyn.val_frac=0"}, {}
    ensemble.eval()
    metrics = {key: [] for key in ("mse", "normalized_mse", "disagreement",
                                   "normalized_disagreement", "aleatoric_variance", "member_nll")}
    scale = ensemble.members[0].delta_std
    for start in range(0, len(val_idx), batch_size):
        idx = val_idx[start:start + batch_size]
        outputs = [member(obs[idx], actions[idx]) for member in ensemble.members]
        means = torch.stack([output[0] for output in outputs])
        variances = torch.stack([torch.exp(2 * output[1]) for output in outputs])
        error = means.mean(dim=0) - (next_obs[idx] - obs[idx])
        disagreement = means.var(dim=0, unbiased=False)
        batch = {
            "mse": error.square().mean(dim=-1),
            "normalized_mse": (error / scale).square().mean(dim=-1),
            "disagreement": disagreement.mean(dim=-1),
            "normalized_disagreement": (disagreement / scale.square()).mean(dim=-1),
            "aleatoric_variance": variances.mean(dim=(0, 2)),
            "member_nll": torch.stack([member.nll_per_sample(obs[idx], actions[idx], next_obs[idx])
                                        for member in ensemble.members], dim=-1),
        }
        for key, value in batch.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite diagnostic: {key}")
            metrics[key].append(value.cpu().numpy())
    metrics = {key: np.concatenate(value) for key, value in metrics.items()}
    ids = trajectory_ids[val_idx.cpu().numpy()]
    val_sources = sources[val_idx.cpu().numpy()]
    report = {"status": "ok", "evaluation": "one_step_on_selection_holdout_after_best_member_restore",
              "overall": _metric_summary(metrics, np.ones(len(ids), dtype=bool), bins),
              "by_source": {}, "by_trajectory": {}}
    for source in np.unique(sources):
        mask = val_sources == source
        report["by_source"][str(source)] = (_metric_summary(metrics, mask, bins) if mask.any()
                                            else {"transitions": 0, "status": "no_holdout_trajectories"})
    for trajectory in np.unique(ids):
        mask = ids == trajectory
        report["by_trajectory"][str(trajectory)] = _metric_summary(metrics, mask, bins)
    report["trajectory_macro_normalized_mse"] = float(np.mean(
        [row["normalized_mse"] for row in report["by_trajectory"].values()]))
    return report, dict(metrics, trajectory_ids=ids, sources=val_sources)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Delay simulator/agent imports so numerical helpers can be tested on CPU alone.
    from agent.dynamics_ensemble import DynamicsEnsemble
    from make_envs import make_env

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env(cfg)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        original_obs_dim = getattr(env, "original_obs_dim", obs_dim)
    finally:
        env.close()
    # Dynamics training bypasses make_agent, which normally fills these fields.
    cfg.agent.obs_dim = obs_dim
    cfg.agent.action_dim = action_dim
    # Resolve before training so invalid metadata cannot discard a completed run.
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    dyn_cfg = cfg.get("dyn", {}) or {}
    effective_obs_dim = int(dyn_cfg.get("effective_obs_dim", obs_dim))
    N = int(getattr(cfg.method, "penalty_N", dyn_cfg.get("N", 5)))
    if not 1 <= effective_obs_dim <= obs_dim or N < 1:
        raise ValueError("Invalid effective_obs_dim or ensemble size")
    obs, actions, next_obs, trajectory_ids, sources = _build_dataset(
        cfg, cfg.seed, obs_dim, action_dim, original_obs_dim)
    obs = obs[:, :effective_obs_dim]
    next_obs = next_obs[:, :effective_obs_dim]
    train_idx, val_idx = _split_trajectories(trajectory_ids, sources, float(dyn_cfg.get("val_frac", .05)), cfg.seed)
    train_np, val_np = train_idx, val_idx
    obs_t, act_t, next_t = (torch.as_tensor(array, dtype=torch.float32, device=cfg.device)
                            for array in (obs, actions, next_obs))
    train_idx, val_idx = (torch.as_tensor(idx, device=cfg.device) for idx in (train_idx, val_idx))
    ensemble = DynamicsEnsemble(obs_dim, action_dim, N=N,
                                hidden_dim=int(dyn_cfg.get("hidden_dim", 256)),
                                hidden_depth=int(dyn_cfg.get("hidden_depth", 3)),
                                effective_obs_dim=effective_obs_dim).to(cfg.device)
    print(f"--> train={len(train_idx)} val={len(val_idx)} transitions | "
          f"train={len(np.unique(trajectory_ids[train_np]))} val={len(np.unique(trajectory_ids[val_np]))} trajectories")
    training = _train_ensemble(ensemble, obs_t, act_t, next_t, train_idx, val_idx, dyn_cfg, cfg.seed)
    diagnostics, samples = _diagnostics(ensemble, obs_t, act_t, next_t, val_idx, trajectory_ids, sources,
                                        int(dyn_cfg.get("batch_size", 256)), int(dyn_cfg.get("diagnostic_bins", 10)))
    training["split"] = {"unit": "trajectory", "val_frac": float(dyn_cfg.get("val_frac", .05)),
                         "train_trajectories": np.unique(trajectory_ids[train_np]).tolist(),
                         "val_trajectories": np.unique(trajectory_ids[val_np]).tolist()}
    training["config"] = resolved_config
    demo_stem = os.path.splitext(os.path.basename(cfg.env.demo))[0]
    save_dir = hydra.utils.to_absolute_path(f"dynamics/{demo_stem}")
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, f"ensemble_{N}")
    ensemble.save(stem + ".pt", training_metadata=training)
    with open(stem + "_diagnostics.json", "w") as stream:
        json.dump({"training": training, "validation": diagnostics}, stream, indent=2, allow_nan=False)
    np.savez_compressed(stem + "_validation.npz", train_indices=train_np, val_indices=val_np, **samples)
    if diagnostics["status"] == "ok":
        print(f"--> Validation diagnostics: {json.dumps(diagnostics['overall'])}")
    print(f"--> Saved {stem}.pt, {stem}_diagnostics.json and {stem}_validation.npz")


if __name__ == "__main__":
    main()
