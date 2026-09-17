"""Offline checks for trajectory grouping, inference and final-run diagnostics."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from test_q_distribution import (
    equal_trajectory_histogram, finalize_q_diagnostics, group_by_return,
    infer_trajectory_q, load_trajectories, progress_statistics,
    run_q_diagnostics, select_examples,
)


class DoubleQCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.child = torch.nn.Identity()

    def forward(self, states, actions):
        first = states[:, :1] * self.scale
        second = states[:, :1] + actions[:, :1]
        return torch.minimum(first, second)


class RecordingRun:
    id = "test-run"

    def __init__(self):
        self.summary = {}
        self.records = []

    def log(self, values):
        self.records.append(values)


def config():
    return OmegaConf.create({
        "agent": {"obs_dim": 2, "action_dim": 1},
        "env": {"name": "HalfCheetah-v2"},
        "q_eval": {"enabled": True, "low_quantile": 0.2, "high_quantile": 0.8,
                   "examples_per_group": 3, "seed": 2026,
                   "batch_size": 2, "progress_points": 101},
    })


def dataset(path, constant_returns=False):
    data = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones", "lengths")}
    for index in range(20):
        length = index + 1
        states = np.column_stack((np.arange(length) + index, np.ones(length))).astype(np.float32)
        data["states"].append(states)
        data["next_states"].append(states + 1)
        data["actions"].append(np.full((length, 1), -0.5, dtype=np.float32))
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1 if constant_returns else index
        data["rewards"].append(rewards)
        data["dones"].append(np.zeros(length, dtype=np.float32))
        data["lengths"].append(length)
    path.write_bytes(pickle.dumps(data))
    return data


def test_quantiles_keep_ties_and_skip_overlap():
    low, high, groups = group_by_return([0, 0, 0, 1, 2, 3, 4, 4, 4, 4])
    assert (low, high) == (0, 4)
    assert groups == {"low": [0, 1, 2], "high": [6, 7, 8, 9]}
    assert group_by_return([1] * 20)[2] == {"low": [], "high": []}
    with pytest.raises(ValueError):
        group_by_return([1, 2], 0.8, 0.2)


def test_examples_are_stable_and_do_not_change_global_rng():
    groups = {"low": list(range(20)), "high": list(range(20, 40))}
    state = np.random.get_state()
    examples = select_examples(groups, 2026, 3)
    assert examples == select_examples(groups, 2026, 3)
    assert np.array_equal(state[1], np.random.get_state()[1])
    assert select_examples({"low": [0], "high": [1]}, 1, 3) == {"low": [0], "high": [1]}


def test_loader_preserves_timeouts_and_checks_lengths_and_ant_slice(tmp_path):
    path = tmp_path / "data.pkl"
    data = dataset(path)
    trajectories = load_trajectories(path, 2, 1)
    assert len(trajectories) == 20
    assert trajectories[-1]["return"] == 19
    assert trajectories[-1]["last_done"] == 0
    assert load_trajectories(path, 1, 1, reduce_obs_dim=1)[0]["states"].shape == (1, 1)
    data["lengths"][2] += 1
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="length"):
        load_trajectories(path, 2, 1)


def test_batched_q_matches_forward_and_restores_modes_on_error(tmp_path):
    path = tmp_path / "data.pkl"
    dataset(path)
    trajectories = load_trajectories(path, 2, 1)
    critic = DoubleQCritic()
    critic.child.eval()
    before = {key: value.clone() for key, value in critic.state_dict().items()}
    values = infer_trajectory_q(critic, trajectories, [0, 19], "cpu", 3)
    expected = np.minimum(trajectories[19]["states"][:, 0] * 2,
                          trajectories[19]["states"][:, 0] - 0.5)
    np.testing.assert_allclose(values[19], expected)
    assert critic.training and not critic.child.training
    assert critic.scale.grad is None
    assert all(torch.equal(before[k], v) for k, v in critic.state_dict().items())
    with patch.object(critic, "forward", side_effect=RuntimeError("broken")):
        with pytest.raises(RuntimeError, match="broken"):
            infer_trajectory_q(critic, trajectories, [0], "cpu", 3)
    assert critic.training and not critic.child.training
    with patch.object(critic, "forward", return_value=torch.tensor([[float("nan")]])):
        with pytest.raises(ValueError, match="Non-finite Q"):
            infer_trajectory_q(critic, trajectories, [0], "cpu", 3)


def test_average_and_distribution_give_each_trajectory_equal_weight():
    curves = [np.zeros(1), np.full(100, 10.0)]
    _, mean, quantiles = progress_statistics(curves, 101)
    np.testing.assert_allclose(mean, 5)
    np.testing.assert_allclose(quantiles[0], 2.5)
    np.testing.assert_allclose(quantiles[1], 7.5)
    np.testing.assert_allclose(equal_trajectory_histogram(curves, [-1, 5, 11]), [0.5, 0.5])


def test_finalizer_saves_checkpoint_and_logs_only_three_images(tmp_path):
    path = tmp_path / "data.pkl"
    dataset(path)
    agent = SimpleNamespace(critic=DoubleQCritic(), device="cpu")
    run = RecordingRun()
    output = tmp_path / "result"
    result = finalize_q_diagnostics(agent, config(), path, run, output)
    assert result["status"] == "logged"
    assert len(run.records) == 1
    assert set(run.records[0]) == {
        "q_eval/example_trajectories", "q_eval/mean_q_by_return",
        "q_eval/q_distribution_by_return",
    }
    for image_path in result["plots"].values():
        assert Path(image_path).read_bytes().startswith(b"\x89PNG")
    assert (output / "config.yaml").is_file()
    state = torch.load(output / "final_critic.pt", weights_only=True)
    assert torch.equal(state["scale"], agent.critic.scale)
    assert json.loads((output / "metadata.json").read_text())["run_id"] == run.id
    assert result["q_definition"] == "Q = min(Q1, Q2)"


def test_constant_returns_skip_and_failure_preserves_checkpoint(tmp_path):
    path = tmp_path / "data.pkl"
    dataset(path, constant_returns=True)
    agent = SimpleNamespace(critic=DoubleQCritic(), device="cpu")
    run = RecordingRun()
    result = run_q_diagnostics(agent, config(), path, run, tmp_path / "skip")
    assert result["status"] == "skipped"
    assert not run.records
    with patch("test_q_distribution.run_q_diagnostics", side_effect=RuntimeError("render failed")):
        result = finalize_q_diagnostics(agent, config(), path, run, tmp_path / "failed")
    assert result["status"] == "failed"
    assert run.summary["q_eval/status"] == "failed"
    assert (tmp_path / "failed" / "final_critic.pt").is_file()
    assert "render failed" in (tmp_path / "failed" / "error.txt").read_text()
    assert json.loads((tmp_path / "failed" / "metadata.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("fail_training,enabled", [(False, True), (True, True), (False, False)])
def test_training_entry_runs_diagnostics_once_then_closes_logs(tmp_path, fail_training, enabled):
    # Execute the actual main function with lightweight environment/optimizer
    # substitutes, so lifecycle ordering is checked without requiring MuJoCo.
    import ast
    import datetime
    import os
    import random
    import threading
    import types
    from unittest.mock import Mock

    path = tmp_path / "data.pkl"
    dataset(path)
    cfg = OmegaConf.merge(config(), {
        "device": "cpu", "seed": 0, "cuda_deterministic": False,
        "project_name": "test", "exp_name": "test", "pretrain": None,
        "method": {}, "log_dir": str(tmp_path), "log_interval": 1,
        "expert": {"demos": 1, "subsample_freq": 1},
        "eval": {"eps": 1, "stochastic": False},
        "agent": {"name": "sac"}, "q_eval": {"enabled": enabled},
        "env": {"replay_mem": 100, "learn_steps": 2, "eval_interval": 1,
                "expert_path": str(path), "supplement_path": str(path)},
    })
    events = []
    run = Mock(id="lifecycle-run")
    run.finish.side_effect = lambda **kw: events.append(("finish", kw["exit_code"]))
    writer = Mock()
    writer.close.side_effect = lambda: events.append("close_writer")
    env = Mock(observation_space=SimpleNamespace(shape=(2,)),
               action_space=SimpleNamespace(shape=(1,)))
    memory = Mock()
    memory.size.return_value = 20
    agent = SimpleNamespace()

    def update(self, policy, expert, logger, step):
        events.append(("update", step))
        if fail_training:
            raise RuntimeError("training failed")
        return {}

    source = Path(__file__).resolve().parents[1] / "train_iq_noisy_expert.py"
    main_node = next(node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_node.decorator_list = []
    namespace = {
        "DictConfig": object, "OmegaConf": OmegaConf, "get_args": lambda x: x,
        "random": random, "np": np, "torch": torch, "os": os,
        "datetime": datetime, "threading": threading, "types": types,
        "make_env": lambda args: env, "make_agent": lambda env, args: agent,
        "wandb": SimpleNamespace(init=lambda **kw: run),
        "hydra": SimpleNamespace(utils=SimpleNamespace(to_absolute_path=str)),
        "Memory": lambda *a, **kw: memory,
        "SummaryWriter": lambda **kw: writer, "Logger": Mock(),
        "iq_update": update, "iq_update_critic": lambda *a: None,
        "tqdm": lambda x: x, "evaluate": lambda *a, **kw: ([1], [1]),
    }
    exec(compile(ast.Module(body=[main_node], type_ignores=[]), str(source), "exec"), namespace)
    with patch("test_q_distribution.finalize_q_diagnostics",
               side_effect=lambda *a: events.append("diagnostics")):
        if fail_training:
            with pytest.raises(RuntimeError, match="training failed"):
                namespace["main"](cfg)
            assert events == [("update", 0), "close_writer", ("finish", 1)]
        else:
            namespace["main"](cfg)
            expected = [("update", 0), ("update", 1)]
            if enabled:
                expected.append("diagnostics")
            assert events == expected + ["close_writer", ("finish", 0)]
