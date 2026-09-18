"""Final offline Q diagnostics; importing this module never starts a run.

The training entry point passes its existing W&B run. For a local rerun:
    python test_q_distribution.py --config CONFIG --critic CRITIC \
        --dataset DATA.pkl --output OUTPUT
Only load trusted pickle and checkpoint files.
"""

import argparse
import json
import pickle
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_trajectories(path, obs_dim, action_dim, reduce_obs_dim=None):
    """Preserve source indices and stored episode boundaries, including timeouts."""
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    fields = ("states", "next_states", "actions", "rewards", "dones", "lengths")
    if not isinstance(data, dict) or any(key not in data for key in fields):
        raise ValueError(f"Expected trajectory dictionary with fields {fields}")
    count = len(data["lengths"])
    if not count or any(len(data[key]) != count for key in fields):
        raise ValueError("Empty data or inconsistent trajectory counts")
    trajectories = []
    for index in range(count):
        raw_length = data["lengths"][index]
        length = int(raw_length)
        if length <= 0 or length != raw_length:
            raise ValueError(f"Trajectory {index}: invalid length {raw_length}")
        arrays = {key: np.asarray(data[key][index]) for key in fields[:-1]}
        for key, values in arrays.items():
            if values.ndim == 0 or len(values) != length:
                raise ValueError(f"Trajectory {index}: {key} does not match length")
            if not np.isfinite(values).all():
                raise ValueError(f"Trajectory {index}: non-finite {key}")
        for key in ("rewards", "dones"):
            if arrays[key].shape not in ((length,), (length, 1)):
                raise ValueError(f"Trajectory {index}: invalid {key} shape")
        for key in ("states", "next_states"):
            values = arrays[key]
            if values.ndim != 2:
                raise ValueError(f"Trajectory {index}: {key} must be a matrix")
            if reduce_obs_dim is not None:
                values = values[:, :reduce_obs_dim]
            if values.shape != (length, obs_dim):
                raise ValueError(f"Trajectory {index}: {key} dimension mismatch")
            arrays[key] = values
        if arrays["actions"].shape != (length, action_dim):
            raise ValueError(f"Trajectory {index}: action dimension mismatch")
        states = arrays["states"].astype(np.float32, copy=False)
        actions = arrays["actions"].astype(np.float32, copy=False)
        total_return = float(arrays["rewards"].sum(dtype=np.float64))
        if not (np.isfinite(states).all() and np.isfinite(actions).all()
                and np.isfinite(total_return)):
            raise ValueError(f"Trajectory {index}: overflow converting inputs or return")
        trajectories.append({
            "id": index, "states": states, "actions": actions,
            "length": length, "return": total_return,
            "last_done": float(arrays["dones"].reshape(-1)[-1]),
        })
    return trajectories


def group_by_return(returns, low_quantile=0.2, high_quantile=0.8):
    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("Require 0 < low_quantile < high_quantile < 1")
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Returns must be a nonempty finite vector")
    low, high = np.quantile(values, [low_quantile, high_quantile])
    if low >= high:
        return float(low), float(high), {"low": [], "high": []}
    return float(low), float(high), {
        "low": np.flatnonzero(values <= low).tolist(),
        "high": np.flatnonzero(values >= high).tolist(),
    }


def select_examples(groups, seed, count):
    if count < 1:
        raise ValueError("examples_per_group must be positive")
    rng = np.random.default_rng(seed)
    return {name: sorted(rng.choice(sorted(ids), min(count, len(ids)),
                                   replace=False).tolist())
            for name, ids in groups.items()}


def infer_trajectory_q(critic, trajectories, ids, device, batch_size, normalizer=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    previous_modes = [(module, module.training) for module in critic.modules()]
    result = {}
    try:
        critic.eval()
        with torch.inference_mode():
            for index in ids:
                trajectory = trajectories[index]
                chunks = []
                for start in range(0, trajectory["length"], batch_size):
                    stop = min(start + batch_size, trajectory["length"])
                    states = torch.as_tensor(trajectory["states"][start:stop], device=device)
                    actions = torch.as_tensor(trajectory["actions"][start:stop], device=device)
                    if normalizer is not None:
                        states = normalizer.normalize_tensor(states)
                    # Use the public forward path, including any configured tanh scaling.
                    q = critic(states, actions)
                    if not torch.is_tensor(q) or q.shape not in ((stop-start,), (stop-start, 1)):
                        raise ValueError("Critic must return one scalar Q per state-action")
                    if not torch.isfinite(q).all():
                        raise ValueError(f"Non-finite Q on trajectory {index}")
                    chunks.append(q.reshape(-1).cpu().numpy().copy())
                result[index] = np.concatenate(chunks)
    finally:
        for module, training in previous_modes:
            module.training = training
    return result


def progress_statistics(curves, points):
    if points < 2:
        raise ValueError("progress_points must be at least 2")
    progress = np.linspace(0, 1, points)
    aligned = np.stack([np.interp(progress, np.linspace(0, 1, len(q)), q)
                        for q in curves])
    return progress, aligned.mean(axis=0), np.quantile(aligned, [0.25, 0.75], axis=0)


def equal_trajectory_histogram(curves, bins):
    """Each trajectory contributes 1/N probability mass, regardless of length."""
    return np.mean([np.histogram(q, bins=bins)[0] / len(q) for q in curves], axis=0)


def make_wandb_charts(trajectories, q_values, groups, examples, metadata, points, wandb):
    subtitle = (f"{metadata['q_definition']} | low return <= {metadata['low_threshold']:.3g} "
                f"(n={len(groups['low'])}), high return >= {metadata['high_threshold']:.3g} "
                f"(n={len(groups['high'])})")
    example_rows = []
    for name in ("low", "high"):
        for index in examples[name]:
            trajectory = trajectories[index]
            series = (f"{name.title()} | ID {index} | "
                      f"R={trajectory['return']:.2f} | T={trajectory['length']}")
            example_rows.extend([
                [step, float(q), series, name, index, trajectory["return"]]
                for step, q in enumerate(q_values[index])
            ])
    example_table = wandb.Table(
        columns=["time_step", "q", "series", "return_group", "trajectory_id", "return"],
        data=example_rows)

    progress_rows = []
    for name, ids in groups.items():
        progress, mean, quantiles = progress_statistics([q_values[i] for i in ids], points)
        for progress_value, mean_value, q25, q75 in zip(
                progress * 100, mean, quantiles[0], quantiles[1]):
            progress_rows.extend([
                [float(progress_value), float(mean_value), f"{name.title()} mean", name, "mean"],
                [float(progress_value), float(q25), f"{name.title()} q25", name, "q25"],
                [float(progress_value), float(q75), f"{name.title()} q75", name, "q75"],
            ])
    progress_table = wandb.Table(
        columns=["progress_percent", "q", "series", "return_group", "statistic"],
        data=progress_rows)

    q_min = min(float(q.min()) for q in q_values.values())
    q_max = max(float(q.max()) for q in q_values.values())
    if q_min == q_max:
        delta = max(0.5, abs(q_min) * 0.01)
        q_min, q_max = q_min - delta, q_max + delta
    bins = np.linspace(q_min, q_max, 51)
    distribution_rows = []
    for name, ids in groups.items():
        probability = equal_trajectory_histogram([q_values[i] for i in ids], bins)
        distribution_rows.extend([
            [float((left + right) / 2), float(probability_value), name,
             float(left), float(right)]
            for left, right, probability_value in zip(bins[:-1], bins[1:], probability)
        ])
    distribution_table = wandb.Table(
        columns=["q_bin_center", "probability", "return_group", "bin_left", "bin_right"],
        data=distribution_rows)

    return {
        "q_eval/example_trajectories_interactive": wandb.plot.line(
            example_table, x="time_step", y="q", stroke="series",
            title="Fixed example trajectories | " + subtitle),
        "q_eval/mean_q_by_return_interactive": wandb.plot.line(
            progress_table, x="progress_percent", y="q", stroke="series",
            title="Mean Q by trajectory progress (mean and 25–75% bounds) | " + subtitle),
        "q_eval/q_distribution_by_return_interactive": wandb.plot.line(
            distribution_table, x="q_bin_center", y="probability", stroke="return_group",
            title="Q distribution by trajectory return (equal trajectory weight) | " + subtitle),
    }


def write_metadata(output_dir, metadata):
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def run_q_diagnostics(agent, args, dataset_path, wandb_run, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = args.q_eval
    reduce_dim = (args.env.get("effective_obs_dim", 27)
                  if args.env.name == "Ant-v2" and args.env.get("reduce_obs_dim", False)
                  else None)
    trajectories = load_trajectories(dataset_path, args.agent.obs_dim,
                                    args.agent.action_dim, reduce_dim)
    low, high, groups = group_by_return([t["return"] for t in trajectories],
                                       cfg.low_quantile, cfg.high_quantile)
    critic_name = type(agent.critic).__name__
    q_definition = ("Q = min(Q1, Q2)" if critic_name == "DoubleQCritic"
                    else f"Q = {critic_name}.forward(s, a)")
    metadata = {
        "status": "prepared", "dataset": str(Path(dataset_path).resolve()),
        "run_id": getattr(wandb_run, "id", None), "q_definition": q_definition,
        "config": dict(cfg), "low_threshold": low, "high_threshold": high,
        "groups": groups,
        "trajectories": [{k: t[k] for k in ("id", "length", "return", "last_done")}
                         for t in trajectories],
        "boundary_note": "Returns sum stored rewards. No timeout/completeness metadata; "
                         "stored boundaries are used, including possible partial episodes. "
                         "done=0 is not treated as an incomplete episode.",
        "interpretation": "Environment return groups diagnose IQ Q ordering, not calibration.",
        "observation_normalization": (
            {"enabled": False}
            if getattr(agent, "observation_normalizer", None) is None
            else {
                "enabled": True,
                **agent.observation_normalizer.summary(),
                "metadata": dict(agent.observation_normalizer.metadata),
            }
        ),
    }
    if not groups["low"] or not groups["high"]:
        metadata.update(status="skipped", reason="Return quantile thresholds coincide")
        write_metadata(output_dir, metadata)
        if wandb_run is not None:
            wandb_run.summary["q_eval/status"] = "skipped: return thresholds coincide"
        print("[Q diagnostics] Skipped: return thresholds coincide", flush=True)
        return metadata
    examples = select_examples(groups, cfg.seed, cfg.examples_per_group)
    metadata["examples"] = examples
    write_metadata(output_dir, metadata)
    ids = sorted(set(groups["low"] + groups["high"]))
    print(f"[Q diagnostics] Evaluating {len(ids)} trajectories; thresholds {low:.3f}, {high:.3f}", flush=True)
    q_values = infer_trajectory_q(
        agent.critic, trajectories, ids, agent.device, cfg.batch_size,
        normalizer=getattr(agent, "observation_normalizer", None))
    charts = {}
    if wandb_run is not None:
        import wandb
        charts = make_wandb_charts(
            trajectories, q_values, groups, examples, metadata,
            cfg.progress_points, wandb)
        # Do not pass explicit step: the existing logger increments W&B's internal
        # step separately from training updates. finish() is owned by the caller.
        wandb_run.log(charts)
        wandb_run.summary["q_eval/status"] = "logged"
    metadata["status"] = "logged" if wandb_run is not None else "local_only"
    metadata["charts"] = sorted(charts)
    write_metadata(output_dir, metadata)
    print(f"[Q diagnostics] Saved results to {output_dir}", flush=True)
    return metadata


def finalize_q_diagnostics(agent, args, dataset_path, wandb_run, output_dir):
    """Preserve the final model before attempting diagnostics; don't lose training."""
    from omegaconf import OmegaConf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.critic.state_dict(), output_dir / "final_critic.pt")
    if getattr(agent, "observation_normalizer", None) is not None:
        agent.observation_normalizer.save(output_dir / "observation_normalizer.npz")
    OmegaConf.save(args, output_dir / "config.yaml", resolve=True)
    try:
        return run_q_diagnostics(agent, args, dataset_path, wandb_run, output_dir)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        metadata_path = output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update(status="failed", error=error)
        write_metadata(output_dir, metadata)
        if wandb_run is not None:
            wandb_run.summary["q_eval/status"] = "failed"
            wandb_run.summary["q_eval/error"] = error
        print(f"[Q diagnostics] Failed; final critic preserved: {error}", flush=True)
        return {"status": "failed", "error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "critic", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()
    import hydra
    from omegaconf import OmegaConf

    args = OmegaConf.load(options.config)
    args.device = options.device
    critic = hydra.utils.instantiate(args.agent.critic_cfg, args=args, _recursive_=False)
    critic.load_state_dict(torch.load(options.critic, map_location="cpu", weights_only=True))
    normalizer = None
    obs_norm_cfg = getattr(args, "observation_normalization", None)
    if obs_norm_cfg is not None and bool(getattr(obs_norm_cfg, "enabled", False)):
        from utils.observation_normalizer import ObservationNormalizer
        stats_path = getattr(obs_norm_cfg, "stats_path", None)
        if stats_path is None:
            candidate = Path(options.config).resolve().parent / "observation_normalizer.npz"
            if not candidate.is_file():
                raise FileNotFoundError(
                    "Normalized Q diagnostics require observation_normalization.stats_path "
                    "or observation_normalizer.npz beside the config")
            stats_path = candidate
        normalizer = ObservationNormalizer.load(stats_path)
    agent = SimpleNamespace(
        critic=critic.to(options.device), device=options.device,
        observation_normalizer=normalizer)
    run_q_diagnostics(agent, args, options.dataset, None, options.output)


if __name__ == "__main__":
    main()
