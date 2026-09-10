"""CPU coverage for dynamics fitting, trajectory holdout and checkpoint inference."""
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

import train_dynamics as training

spec = importlib.util.spec_from_file_location(
    'dynamics_model_test', Path(__file__).resolve().parents[1] / 'agent/dynamics_ensemble.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class DynamicsTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_normalization_and_original_unit_sampling(self):
        ensemble = model.DynamicsEnsemble(3, 1, N=1, effective_obs_dim=2, hidden_depth=0)
        member = ensemble.members[0]
        obs = torch.tensor([[1., 8.], [5., 8.]])
        actions = torch.tensor([[2.], [6.]])
        next_obs = obs + torch.tensor([[2., 3.], [6., 3.]])
        member.fit_normalization(obs, actions, next_obs)
        torch.testing.assert_close(member.obs_mean, torch.tensor([3., 8.]))
        torch.testing.assert_close(member.obs_std, torch.tensor([2., 1.]))
        torch.testing.assert_close(member.delta_std, torch.tensor([2., 1.]))
        with torch.no_grad():
            member.trunk[0].weight.zero_()
            member.trunk[0].bias.zero_()
            member.trunk[0].weight[0, 0] = 1
            member.trunk[0].weight[1, 2] = 1
        normalized_mean, normalized_log_std = member._forward_raw(obs, actions)
        torch.testing.assert_close(normalized_mean, torch.tensor([[-1., -1.], [1., 1.]]))
        mean, log_std = member(obs, actions)
        torch.testing.assert_close(mean, normalized_mean * member.delta_std + member.delta_mean)
        torch.testing.assert_close(log_std, normalized_log_std + member.delta_std.log())
        full_obs = torch.cat([obs, torch.full((2, 1), 99.)], dim=-1)
        samples = ensemble.sample_next_ensemble(full_obs, actions, noise=torch.ones(2, 1, 2))
        torch.testing.assert_close(samples[:, 0, 0, :2], obs + mean + log_std.exp())
        torch.testing.assert_close(samples[:, 0, 0, 2], full_obs[:, 2])
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            torch.testing.assert_close(member.sample_next(obs, actions), samples[:, 0, 0, :2])

    def test_checkpoint_roundtrip_legacy_and_corruption(self):
        source = model.DynamicsEnsemble(2, 1, N=2, hidden_dim=7, hidden_depth=1)
        obs, act = torch.randn(6, 2), torch.randn(6, 1)
        for member in source.members:
            member.fit_normalization(obs, act, obs * 3 + 4)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.pt')
            source.save(path)
            target = model.DynamicsEnsemble(2, 1, N=2, hidden_dim=7, hidden_depth=1)
            target.load(path)
            noise = torch.randn(6, 2, 2)
            torch.testing.assert_close(source.sample_next_ensemble(obs, act, 2, noise),
                                       target.sample_next_ensemble(obs, act, 2, noise))
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['cfg']['hidden_dim'], 7)
            del payload['state_dict']['members.0.obs_mean']
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                target.load(path)
            identity = model.DynamicsEnsemble(2, 1, N=2, hidden_dim=7, hidden_depth=1)
            old_state = {k: v for k, v in identity.state_dict().items()
                         if not k.endswith(('_mean', '_std')) or k.endswith(('max_log_std', 'min_log_std'))}
            for old_payload in (old_state, {'state_dict': old_state, 'cfg': {'obs_dim': 2, 'action_dim': 1, 'N': 2}}):
                torch.save(old_payload, path)
                target.load(path)
                torch.testing.assert_close(target(0, obs, act), identity(0, obs, act))

    def test_trajectory_split_no_leakage_and_singletons(self):
        ids = np.array(['expert:0'] * 4 + ['supplement:0'] * 2 + ['supplement:1'] * 3)
        sources = np.array(['expert'] * 4 + ['supplement'] * 5)
        with self.assertWarns(UserWarning):
            train, val = training._split_trajectories(ids, sources, .05, 42)
        self.assertFalse(set(ids[train]) & set(ids[val]))
        self.assertIn('expert:0', ids[train])
        self.assertEqual(len(train) + len(val), len(ids))
        with self.assertWarns(UserWarning):
            repeat = training._split_trajectories(ids, sources, .05, 42)
        np.testing.assert_array_equal(val, repeat[1])
        with self.assertWarns(UserWarning), self.assertRaises(ValueError):
            training._split_trajectories(ids[:4], sources[:4], .1, 0)
        self.assertEqual(len(training._split_trajectories(ids, sources, 0, 0)[1]), 0)
        for frac in (-.1, 1, float('nan')):
            with self.assertRaises(ValueError):
                training._split_trajectories(ids, sources, frac, 0)

    def test_fixed_bootstrap_and_independent_best_epochs(self):
        ensemble = model.DynamicsEnsemble(1, 1, N=2, hidden_dim=8, hidden_depth=1)
        obs = torch.arange(10.).reshape(-1, 1)
        act, nxt = obs / 10, obs + .1 * obs.square()
        train_idx, val_idx = torch.arange(8), torch.arange(8, 10)
        seen = [[], []]
        originals = [member.nll_loss for member in ensemble.members]
        def recorder(i):
            def call(o, a, n):
                seen[i].extend(o[:, 0].tolist())
                return originals[i](o, a, n)
            return call
        snapshots = []
        def validation(*args):
            snapshots.append([{key: val.clone() for key, val in member.state_dict().items()}
                              for member in ensemble.members])
            return np.array([[1., 3.], [2., 1.], [3., 2.]][len(snapshots) - 1])
        cfg = {'epochs': 3, 'batch_size': 3, 'log_interval': 3}
        with patch.object(ensemble.members[0], 'nll_loss', side_effect=recorder(0)), \
                patch.object(ensemble.members[1], 'nll_loss', side_effect=recorder(1)), \
                patch.object(training, '_validation_losses', side_effect=validation):
            result = training._train_ensemble(ensemble, obs, act, nxt, train_idx, val_idx, cfg, 11)
        self.assertEqual(result['best_epochs'], [1, 2])
        for i, best_epoch in enumerate([0, 1]):
            for key, value in ensemble.members[i].state_dict().items():
                torch.testing.assert_close(value, snapshots[best_epoch][i][key])
            chunks = [sorted(seen[i][start:start + 8]) for start in range(0, 24, 8)]
            self.assertEqual(chunks[0], chunks[1])
            self.assertEqual(chunks[0], chunks[2])
            self.assertLess(max(seen[i]), 8)
            torch.testing.assert_close(ensemble.members[i].obs_mean, obs[:8].mean(0))
        draws = training._fixed_bootstrap(train_idx, 2, 1011)
        self.assertFalse(torch.equal(draws[0], draws[1]))
        for i in range(2):
            self.assertEqual(sorted(draws[i].tolist()), sorted(seen[i][:8]))

    def test_diagnostics_known_error_disagreement_and_batching(self):
        ensemble = model.DynamicsEnsemble(1, 1, N=2, hidden_depth=0)
        with torch.no_grad():
            for i, member in enumerate(ensemble.members):
                member.trunk[0].weight.zero_()
                member.trunk[0].bias.zero_()
                member.trunk[0].weight[0, 0] = i * 2.
        obs = torch.arange(1., 5.).reshape(-1, 1)
        act, idx = torch.zeros(4, 1), torch.arange(4)
        ids, sources = np.array(['s:0', 's:0', 's:1', 's:1']), np.array(['s'] * 4)
        report, samples = training._diagnostics(ensemble, obs, act, obs, idx, ids, sources, 3)
        np.testing.assert_allclose(samples['mse'], [1, 4, 9, 16])
        np.testing.assert_allclose(samples['disagreement'], [1, 4, 9, 16])
        self.assertAlmostEqual(report['overall']['error_disagreement_pearson'], 1.)
        self.assertAlmostEqual(report['overall']['error_disagreement_spearman'], 1.)
        self.assertEqual(len(report['by_trajectory']), 2)
        _, repeat = training._diagnostics(ensemble, obs, act, obs, idx, ids, sources, 1)
        for key in ('mse', 'member_nll', 'disagreement'):
            np.testing.assert_allclose(samples[key], repeat[key])
        ensemble.members[1].load_state_dict(ensemble.members[0].state_dict())
        report, samples = training._diagnostics(ensemble, obs, act, obs + 1, idx, ids, sources, 2)
        np.testing.assert_array_equal(samples['disagreement'], np.zeros(4))
        self.assertIsNone(report['overall']['error_disagreement_pearson'])
        self.assertEqual(len(report['overall']['disagreement_bins']), 1)
        json.dumps(report, allow_nan=False)

    def test_training_entrypoint_writes_reloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dict(states=[], next_states=[], actions=[], rewards=[], dones=[], lengths=[])
            for i in range(4):
                obs = np.arange(12, dtype=np.float32).reshape(6, 2) / 10 + i
                for key, value in dict(states=obs, next_states=obs + .2, actions=obs[:, :1] / 10,
                                       rewards=np.zeros(6), dones=np.zeros(6), lengths=6).items():
                    data[key].append(value)
            for name in ('expert', 'supplement'):
                with (root / (name + '.pkl')).open('wb') as stream:
                    pickle.dump(data, stream)
            cfg = OmegaConf.create({'device': 'cpu', 'seed': 2,
                'env': {'expert_path': 'expert.pkl', 'supplement_path': 'supplement.pkl', 'demo': 'smoke.pkl'},
                'expert': {'demos': 2, 'subsample_freq': 1}, 'method': {'penalty_N': 2},
                'dyn': {'epochs': 2, 'batch_size': 7, 'hidden_dim': 8, 'hidden_depth': 1, 'val_frac': .25}})
            env = SimpleNamespace(observation_space=SimpleNamespace(shape=(2,)),
                                  action_space=SimpleNamespace(shape=(1,)), close=lambda: None)
            with patch.dict('sys.modules', {'agent.dynamics_ensemble': model,
                                           'make_envs': SimpleNamespace(make_env=lambda cfg: env)}), \
                    patch.object(training.hydra.utils, 'to_absolute_path', side_effect=lambda p: str(root / p)):
                training.main.__wrapped__(cfg)
            stem = root / 'dynamics/smoke/ensemble_2'
            with Path(str(stem) + '_diagnostics.json').open() as stream:
                report = json.load(stream)
            self.assertEqual(report['validation']['status'], 'ok')
            loaded = model.DynamicsEnsemble(2, 1, N=2, hidden_dim=8, hidden_depth=1)
            loaded.load(str(stem) + '.pt')
            self.assertEqual(len(report['training']['best_epochs']), 2)
            with np.load(str(stem) + '_validation.npz') as samples:
                self.assertEqual(len(samples['val_indices']), len(samples['mse']))
                self.assertFalse(set(samples['train_indices']) & set(samples['val_indices']))


if __name__ == '__main__':
    unittest.main()
