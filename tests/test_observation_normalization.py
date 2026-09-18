import numpy as np
import torch

from utils.observation_normalizer import (
    NormalizedReplayView,
    ObservationNormalizer,
    validate_agent_observation_normalizer,
)
from wrappers.normalize_observation_wrapper import NormalizeObservationWrapper


class Replay:
    def __init__(self):
        self.state = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
        self.next_state = self.state + torch.tensor([1.0, 2.0])

    def size(self):
        return 2

    def raw_observations(self):
        return self.state.numpy()

    def get_samples(self, batch_size, device):
        return (self.state.to(device), self.next_state.to(device),
                torch.ones(2, 1, device=device), torch.zeros(2, 1, device=device),
                torch.zeros(2, 1, device=device))


class Env:
    def __init__(self):
        self.value = np.array([1.0, 4.0], dtype=np.float32)

    def reset(self):
        return self.value.copy()

    def step(self, action):
        return self.value + 1, 2.0, True, {"episode": {"r": 2.0, "l": 1}}


def test_formula_roundtrip_and_persistence(tmp_path):
    observations = np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    normalizer = ObservationNormalizer.fit(observations, eps=1e-3, metadata={"source": "supplement"})
    expected = (observations - observations.mean(0)) / (observations.std(0) + 1e-3)
    np.testing.assert_allclose(normalizer.normalize_np(observations), expected)
    np.testing.assert_allclose(
        normalizer.denormalize_np(normalizer.normalize_np(observations)), observations,
        rtol=1e-6, atol=1e-6)

    tensor = torch.tensor(observations).view(1, 1, 2, 2)
    torch.testing.assert_close(
        normalizer.denormalize_tensor(normalizer.normalize_tensor(tensor)), tensor)

    path = tmp_path / "normalizer.npz"
    normalizer.save(path)
    loaded = ObservationNormalizer.load(path)
    normalizer.assert_compatible(loaded)
    assert loaded.metadata == {"source": "supplement"}


def test_constant_dimension_matches_imitation_dice_formula():
    observations = np.array([[2.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    normalizer = ObservationNormalizer.fit(observations, eps=1e-3)
    assert np.isclose(normalizer.scale[0], 1000.0)
    np.testing.assert_array_equal(normalizer.normalize_np(observations)[:, 0], 0.0)


def test_replay_and_environment_share_one_transform_without_mutating_raw_data():
    replay = Replay()
    raw_before = replay.state.clone()
    normalizer = ObservationNormalizer.fit(replay.raw_observations())
    view = NormalizedReplayView(replay, normalizer)
    state, next_state, action, reward, done = view.get_samples(2, "cpu")
    torch.testing.assert_close(state, normalizer.normalize_tensor(raw_before))
    torch.testing.assert_close(next_state, normalizer.normalize_tensor(replay.next_state))
    torch.testing.assert_close(replay.state, raw_before)
    torch.testing.assert_close(action, torch.ones(2, 1))
    torch.testing.assert_close(reward, torch.zeros(2, 1))
    torch.testing.assert_close(done, torch.zeros(2, 1))

    env = NormalizeObservationWrapper(Env(), normalizer)
    np.testing.assert_allclose(env.reset(), state[0].numpy())
    observation, episode_reward, terminal, info = env.step(np.zeros(1))
    np.testing.assert_allclose(observation, normalizer.normalize_np(np.array([2.0, 5.0])))
    assert (episode_reward, terminal, info["episode"]) == (2.0, True, {"r": 2.0, "l": 1})


def test_replay_rejects_repeated_observation_normalization():
    replay = Replay()
    normalizer = ObservationNormalizer.fit(replay.raw_observations())
    normalized_replay = NormalizedReplayView(replay, normalizer)
    try:
        NormalizedReplayView(normalized_replay, normalizer)
    except ValueError as exc:
        assert "already observation-normalized" in str(exc)
    else:
        raise AssertionError("repeated replay normalization was accepted")


def test_environment_rejects_repeated_observation_normalization():
    normalizer = ObservationNormalizer.fit(Replay().raw_observations())
    normalized_env = NormalizeObservationWrapper(Env(), normalizer)
    try:
        NormalizeObservationWrapper(normalized_env, normalizer)
    except ValueError as exc:
        assert "already observation-normalized" in str(exc)
    else:
        raise AssertionError("repeated environment normalization was accepted")


def test_checkpoint_sidecar_rejects_disabled_observation_normalization(tmp_path):
    base_path = tmp_path / "agent"
    normalizer = ObservationNormalizer.fit(Replay().raw_observations())
    normalizer.save(f"{base_path}_obs_normalizer.npz")

    agent = type("Agent", (), {"observation_normalizer": None})()
    try:
        validate_agent_observation_normalizer(agent, str(base_path))
    except ValueError as exc:
        assert "requires normalized observations" in str(exc)
    else:
        raise AssertionError("normalized checkpoint was accepted with normalization disabled")


def test_invalid_shapes_and_nonfinite_values_fail():
    for values in (np.array([]), np.array([1.0, 2.0]), np.array([[np.nan, 0.0]])):
        try:
            ObservationNormalizer.fit(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid normalization input was accepted")
    normalizer = ObservationNormalizer.fit(np.zeros((2, 2), dtype=np.float32))
    try:
        normalizer.normalize_np(np.zeros(3, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("observation dimension mismatch was accepted")
