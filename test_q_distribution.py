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


def infer_trajectory_q(critic, trajectories, ids, device, batch_size):
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


def make_plots(trajectories, q_values, groups, examples, metadata, output_dir, points):
    # Headless rendering without changing the process-wide pyplot backend.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    colors = {"low": "#d97706", "high": "#2563eb"}
    subtitle = (f"{metadata['q_definition']} | low return <= {metadata['low_threshold']:.3g} "
                f"(n={len(groups['low'])}), high return >= {metadata['high_threshold']:.3g} "
                f"(n={len(groups['high'])})")
    paths = {}

    def save(figure, name):
        FigureCanvasAgg(figure)
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        paths[f"q_eval/{name}"] = path
        figure.clear()

    rows = max(len(ids) for ids in examples.values())
    figure = Figure(figsize=(12, 2.7 * rows + 1), constrained_layout=True)
    axes = figure.subplots(rows, 2, squeeze=False, sharey=True)
    figure.suptitle("Fixed example trajectories\n" + subtitle, fontsize=11)
    for col, name in enumerate(("low", "high")):
        for row in range(rows):
            ax = axes[row, col]
            if row >= len(examples[name]):
                ax.set_visible(False)
                continue
            index = examples[name][row]
            trajectory = trajectories[index]
            ax.plot(np.arange(trajectory["length"]), q_values[index], color=colors[name], lw=1,
                    marker="o" if trajectory["length"] == 1 else None)
            ax.set(title=f"{name.title()} | ID {index} | R={trajectory['return']:.2f} | T={trajectory['length']}",
                   xlabel="Time step", ylabel="Q(s, dataset action)")
            ax.grid(alpha=0.2)
    save(figure, "example_trajectories")

    figure = Figure(figsize=(10, 5), constrained_layout=True)
    ax = figure.subplots()
    for name, ids in groups.items():
        progress, mean, quantiles = progress_statistics([q_values[i] for i in ids], points)
        ax.plot(progress * 100, mean, color=colors[name], label=f"{name.title()} return")
        ax.fill_between(progress * 100, *quantiles, color=colors[name], alpha=0.18)
    ax.set(title="Mean Q by trajectory progress\n" + subtitle,
           xlabel="Trajectory progress (%) — shaded: 25–75% across trajectories",
           ylabel="Mean Q (equal trajectory weight)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(figure, "mean_q_by_return")

    figure = Figure(figsize=(10, 5), constrained_layout=True)
    ax = figure.subplots()
    q_min = min(float(q.min()) for q in q_values.values())
    q_max = max(float(q.max()) for q in q_values.values())
    if q_min == q_max:
        delta = max(0.5, abs(q_min) * 0.01)
        q_min, q_max = q_min - delta, q_max + delta
    bins = np.linspace(q_min, q_max, 51)
    for name, ids in groups.items():
        probability = equal_trajectory_histogram([q_values[i] for i in ids], bins)
        ax.stairs(probability, bins, fill=True, alpha=0.35,
                  color=colors[name], label=f"{name.title()} return")
    ax.set(title="Q distribution by trajectory return\n" + subtitle,
           xlabel="Q(s, dataset action)", ylabel="Probability per bin (equal trajectory weight)")
    ax.legend()
    save(figure, "q_distribution_by_return")
    return paths


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
    q_values = infer_trajectory_q(agent.critic, trajectories, ids, agent.device, cfg.batch_size)
    paths = make_plots(trajectories, q_values, groups, examples, metadata,
                       output_dir, cfg.progress_points)
    if wandb_run is not None:
        import wandb
        # Do not pass explicit step: the existing logger increments W&B's internal
        # step separately from training updates. finish() is owned by the caller.
        wandb_run.log({key: wandb.Image(str(path), caption=q_definition)
                       for key, path in paths.items()})
        wandb_run.summary["q_eval/status"] = "logged"
    metadata["status"] = "logged" if wandb_run is not None else "local_only"
    metadata["plots"] = {key: str(path) for key, path in paths.items()}
    write_metadata(output_dir, metadata)
    print(f"[Q diagnostics] Saved results to {output_dir}", flush=True)
    return metadata


def finalize_q_diagnostics(agent, args, dataset_path, wandb_run, output_dir):
    """Preserve the final model before attempting diagnostics; don't lose training."""
    from omegaconf import OmegaConf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.critic.state_dict(), output_dir / "final_critic.pt")
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
    agent = SimpleNamespace(critic=critic.to(options.device), device=options.device)
    run_q_diagnostics(agent, args, options.dataset, None, options.output)


if __name__ == "__main__":
    main()
