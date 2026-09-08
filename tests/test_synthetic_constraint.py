"""Analytical and critic-step checks for the one-step model constraint."""
import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import iq
from test_uncertainty import ROOT, fixture, load_model_file
from utils.utils import average_dicts, get_concat_samples
from utils.robosuite_termination import RobosuiteTermination


class FixedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.tensor(0.2))

    def sample(self, obs, noise=None):
        action = self.action.expand(obs.size(0), 1)
        return action, torch.full_like(action, -0.3), action


def synthetic_fixture():
    agent, batch = fixture((-1., 1.))
    agent.args.method.synthetic_constrain = True
    agent.args.method.synthetic_M = 1
    agent.args.method.synthetic_coef = 0.2
    agent.args.method.synthetic_warmup_steps = 0
    agent.args.env = NS(name='HalfCheetah-v2')
    agent.alpha = torch.tensor(0.5, requires_grad=True)
    agent.actor = FixedActor()
    return agent, batch


class SyntheticConstraintTests(unittest.TestCase):
    def test_disabled_is_noop_including_rng(self):
        agent, batch = fixture()
        rng = torch.random.get_rng_state().clone()
        loss, logs = iq.synthetic_iq_loss(agent, batch[0], 100, True)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logs, {})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_expected_target_penalty_and_critic_only_gradients(self):
        agent, batch = synthetic_fixture()
        obs = batch[0]
        # Deterministic model samples isolate expectation order and entropy.
        states = obs[:, None, None, :].expand(4, 2, agent.args.method.synthetic_M, 2).clone()
        states[:, 0, :, 0] = 0.5
        states[:, 1, :, 0] = 1.5
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states) as sample:
            loss, logs = iq.synthetic_iq_loss(agent, obs, 1, True)
        action = torch.full((4, 1), 0.2)
        torch.testing.assert_close(sample.call_args.args[1], action)
        self.assertFalse(torch.equal(action, batch[2]))
        # E Q^- = 1 + 3*.2, entropy contribution = .5*.3; gamma=.9.
        target = 0.9 * 1.75
        uncertainty = 0.9 * 0.5
        q = obs[:, :1] + 0.6
        residual = torch.stack([q, q + 0.5]) - target + 0.1 * uncertainty
        expected = 0.2 * iq.bellman_constraint(residual, agent.args).mean()
        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(logs['synthetic/uncertainty'], uncertainty, places=6)
        self.assertEqual(logs['synthetic/terminal_fraction'], 0)
        loss.backward()
        self.assertGreater(agent.critic.weight.grad.abs().item(), 0)
        for module in (agent.actor, agent.critic_target, agent.dynamics_ensemble):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNone(agent.alpha.grad)

    def test_average_values_before_nonlinear_constraint(self):
        agent, batch = synthetic_fixture()
        agent.args.method.uncertainty = False
        agent.args.q_net._target_ = 'SingleQ'
        agent.args.left, agent.args.right = -0.1, 0.1
        agent.actor.action.data.zero_()
        agent.alpha = torch.tensor(0.)
        obs = torch.zeros_like(batch[0])
        states = torch.zeros(4, 2, agent.args.method.synthetic_M, 2)
        states[:, 0, :, 0] = -2
        states[:, 1, :, 0] = 2
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states):
            loss, _ = iq.synthetic_iq_loss(agent, obs, 1)
        # The mean target is zero, although every individual target violates.
        self.assertEqual(loss.item(), 0)

        # With uncertainty disabled, member disagreement must not reweight a
        # nonzero violation when the mean Bellman target stays the same.
        obs = torch.ones_like(obs)
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states):
            dispersed_loss, _ = iq.synthetic_iq_loss(agent, obs, 1)
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=torch.zeros_like(states)):
            identical_loss, _ = iq.synthetic_iq_loss(agent, obs, 1)
        self.assertGreater(dispersed_loss.item(), 0)
        torch.testing.assert_close(dispersed_loss, identical_loss)

    def test_model_termination_not_dataset_done(self):
        agent, batch = synthetic_fixture()
        agent.args.env.name = 'Ant-v2'
        agent.args.method.uncertainty = False
        agent.args.q_net._target_ = 'SingleQ'
        states = torch.zeros(4, 2, agent.args.method.synthetic_M, 2)
        states[:, 0, :, 0] = 0.5  # continues
        states[:, 1, :, 0] = 1.5  # terminal
        # Stored transitions all terminate, but synthetic continuation is 50%.
        batch[4].fill_(1)
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states):
            loss, logs = iq.synthetic_iq_loss(agent, batch[0], 1, True)
        q = batch[0][:, :1] + 0.6
        expected_target = 0.9 * (0.5 + 0.6 + 0.15) / 2
        expected = 0.2 * iq.bellman_constraint(q - expected_target, agent.args).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(logs['synthetic/terminal_fraction'], 0.5)

    def test_termination_boundaries(self):
        cases = {
            'Ant-v2': ([[0.19, 0], [0.2, 0], [1., 0], [1.01, 0]], [True, False, False, True]),
            'Hopper-v2': ([[0.7, 0], [0.8, 0.2], [0.8, 0.1]], [True, True, False]),
            'Walker2d-v2': ([[0.8, 0], [2., 0], [1., 1.], [1., 0]], [True, True, True, False]),
            'HalfCheetah-v2': ([[0., 0], [-100., 100]], [False, False]),
        }
        for name, (obs, expected) in cases.items():
            with self.subTest(env=name):
                self.assertEqual(iq.synthetic_done(torch.tensor(obs), name).flatten().tolist(), expected)
        with self.assertRaisesRegex(ValueError, 'no termination rule'):
            iq.synthetic_done(torch.zeros(1, 2), 'Lift')

    def test_warmup_and_shared_resampled_noise(self):
        agent, batch = synthetic_fixture()
        agent.args.method.synthetic_warmup_steps = 100
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble',
                          wraps=agent.dynamics_ensemble.sample_next_ensemble) as sample:
            loss, _ = iq.synthetic_iq_loss(agent, batch[0], 0)
            self.assertEqual(loss.item(), 0)
            sample.assert_not_called()
            for step, coef in ((50, 0.1), (100, 0.2), (200, 0.2)):
                _, logs = iq.synthetic_iq_loss(agent, batch[0], step, True)
                self.assertEqual(logs['synthetic/coef'], coef)
            noises = [call.kwargs['noise'] for call in sample.call_args_list]
            self.assertEqual(noises[0].shape, (4, agent.args.method.synthetic_M, 1))
            self.assertFalse(torch.equal(noises[0], noises[1]))

    def test_invalid_config_and_predictions(self):
        agent, batch = synthetic_fixture()
        iq.validate_synthetic_config(agent)
        agent.args.method.synthetic_M = 0
        with self.assertRaisesRegex(ValueError, 'synthetic_M'):
            iq.validate_synthetic_config(agent)
        agent.args.method.synthetic_M = 1
        agent.args.method.constrain = False
        with self.assertRaises(ValueError):
            iq.validate_synthetic_config(agent)
        agent.args.method.constrain = True
        agent.args.env.name = 'Unknown'
        with self.assertRaises(ValueError):
            iq.validate_synthetic_config(agent)
        agent.args.env.name = 'HalfCheetah-v2'
        states = torch.full((4, 2, agent.args.method.synthetic_M, 2), float('nan'))
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states):
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                iq.synthetic_iq_loss(agent, batch[0], 1)

    def test_synthetic_only_loads_and_freezes_checkpoint(self):
        module = load_model_file('dynamics_ensemble')
        for depth in (0, 2):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as directory:
                agent, _ = synthetic_fixture()
                agent.args.device = 'cpu'
                agent.args.method.uncertainty = False
                agent.args.method.dynamics_ckpt = str(Path(directory) / 'model.pt')
                source = module.DynamicsEnsemble(2, 1, N=2, effective_obs_dim=1,
                                                 hidden_dim=8, hidden_depth=depth)
                source.save(agent.args.method.dynamics_ckpt)
                del agent.dynamics_ensemble
                module.load_iq_dynamics(agent, 2, 1)
                loaded = agent.dynamics_ensemble
                self.assertFalse(loaded.training)
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                self.assertEqual(loaded.effective_obs_dim, 1)
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)

    def test_entry_points_apply_auxiliary_loss_without_changing_dual_statistics(self):
        for filename in ('train_iq.py', 'train_iq_noisy_expert.py', 'train_iq_offline.py',
                         'train_iq_robosuite.py'):
            tree = ast.parse((ROOT / filename).read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'iq_update_critic')
            for double in (False, True):
                with self.subTest(file=filename, double=double):
                    agent, batch = synthetic_fixture()
                    agent.args.q_net._target_ = 'DoubleQ' if double else 'SingleQ'
                    baseline = copy.deepcopy(agent)
                    baseline.args.method.synthetic_constrain = False
                    # fixture getV closures are not used for gradients here.
                    for current in (agent, baseline):
                        current.getV = lambda obs: torch.zeros_like(obs[:, :1])
                        current.get_targetV = lambda obs: torch.zeros_like(obs[:, :1])
                    namespace = dict(torch=torch, iq_loss=iq.iq_loss,
                                     prepare_iq_step=iq.prepare_iq_step,
                                     update_iq_penalty=iq.update_iq_penalty,
                                     synthetic_iq_loss=iq.synthetic_iq_loss,
                                     average_dicts=average_dicts, get_concat_samples=get_concat_samples)
                    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
                    # Compare the loss gradients before the Robosuite entry
                    # point's existing clipping can obscure their difference.
                    with patch('torch.nn.utils.clip_grad_norm_'):
                        for current in (agent, baseline):
                            torch.manual_seed(7)
                            namespace['iq_update_critic'](
                                current, tuple(x[:2] for x in batch[:5]),
                                tuple(x[2:] for x in batch[:5]), Mock(), 1)
                    torch.testing.assert_close(agent.log_penalty.grad, baseline.log_penalty.grad)
                    self.assertEqual(agent.penalty_optimizer.state[agent.log_penalty]['step'].item(), 1)
                    self.assertNotEqual(agent.critic.weight.item(), baseline.critic.weight.item())

    def test_robosuite_lift_layout_and_success_boundary(self):
        # Object is deliberately not the first flattened observation key.
        done_fn = RobosuiteTermination('Lift', {'robot0_eef_pos': (3,), 'object': (10,)})
        obs = torch.zeros(3, 13)
        obs[:, 5] = torch.tensor([0.83, 0.84, 0.85])
        self.assertEqual(done_fn(obs).flatten().tolist(), [False, False, True])
        with self.assertRaisesRegex(ValueError, 'standard single-arm'):
            RobosuiteTermination('Lift', {'object': (14,)})

    def test_robosuite_can_bin_release_and_metadata(self):
        done_fn = RobosuiteTermination(
            'PickPlaceCan', {'robot0_eef_pos': (3,), 'object': (14,)},
            {'bin2_pos': (1., 2., 0.8), 'table_full_size': (0.4, 0.6, 0.82)})
        obs = torch.zeros(4, 17)
        # Can world position is at object[7:10], not object[0:3].
        obs[:, 10:13] = torch.tensor([[1.1, 2.1, 0.85], [1.1, 2.1, 0.85],
                                     [1., 2.1, 0.85], [1.1, 2.1, 0.9]])
        obs[:, :3] = obs[:, 10:13]
        obs[[0, 2, 3], 0] += 0.1
        self.assertEqual(done_fn(obs).flatten().tolist(), [True, False, False, False])
        with self.assertRaisesRegex(ValueError, 'robot0_eef_pos'):
            RobosuiteTermination('PickPlaceCan', {'object': (14,)})

    def test_robosuite_callback_without_mujoco_env_config(self):
        agent, batch = synthetic_fixture()
        del agent.args.env
        agent.synthetic_done_fn = RobosuiteTermination('Lift', {'object': (10,)})
        iq.validate_synthetic_config(agent)
        obs = torch.zeros(4, 10)
        obs[:, 0] = 2.0  # Ensure the auxiliary constraint has a nonzero violation.
        states = torch.zeros(4, 2, agent.args.method.synthetic_M, 10)
        states[:, 0, :, 2] = 0.83
        states[:, 1, :, 2] = 0.85
        with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble', return_value=states):
            loss, logs = iq.synthetic_iq_loss(agent, obs, 1, True)
        self.assertEqual(logs['synthetic/terminal_fraction'], 0.5)
        self.assertGreater(loss.item(), 0)

    def test_logged_warmup_step_has_stable_metric_keys(self):
        agent, batch = synthetic_fixture()
        agent.args.method.synthetic_warmup_steps = 100
        loss, initial = iq.synthetic_iq_loss(agent, batch[0], 0, True)
        _, later = iq.synthetic_iq_loss(agent, batch[0], 100, True)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(initial.keys(), later.keys())

    def test_synthetic_and_real_sampling_counts_are_independent(self):
        for synthetic_m, penalty_m in ((1, 10), (3, 7)):
            with self.subTest(synthetic_M=synthetic_m, penalty_M=penalty_m):
                agent, batch = synthetic_fixture()
                agent.args.method.synthetic_M = synthetic_m
                agent.args.method.penalty_M = penalty_m
                with patch.object(agent.dynamics_ensemble, 'sample_next_ensemble',
                                  wraps=agent.dynamics_ensemble.sample_next_ensemble) as sample:
                    iq.synthetic_iq_loss(agent, batch[0], 1)
                    iq._compute_dynamics_penalty(agent, batch)
                self.assertEqual([call.kwargs['M'] for call in sample.call_args_list],
                                 [synthetic_m, penalty_m])
                self.assertEqual(agent.args.method.penalty_M, penalty_m)


if __name__ == '__main__':
    unittest.main()
