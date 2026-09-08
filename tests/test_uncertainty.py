"""CPU regression tests; no simulator or training entry-point imports required."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import iq
from utils.utils import get_concat_samples, average_dicts

ROOT = Path(__file__).resolve().parents[1]


def load_model_file(name):
    # agent/__init__.py eagerly imports Gym/SAC; these unit tests only need the models.
    spec = importlib.util.spec_from_file_location(name, ROOT / 'agent' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DynamicsEnsemble = load_model_file('dynamics_ensemble').DynamicsEnsemble
DiagGaussianActor = load_model_file('sac_models').DiagGaussianActor


class ConstantDynamics(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, obs, action):
        return self.offset.expand_as(obs), torch.zeros_like(obs)


class Critic(nn.Module):
    def __init__(self, action_sensitive=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.action_sensitive = action_sensitive

    def forward(self, obs, action, both=False):
        q = self.weight * obs[:, :1]
        if self.action_sensitive:
            q = q + 3 * action[:, :1]
        return (q, q + 0.5) if both else q


def fixture(offsets=(0., 0., 0.)):
    method = NS(type="iq", uncertainty=True, penalty_N=len(offsets), penalty_M=10,
                penalty_coef=0.1, constrain=True, penalty_auto=True,
                penalty_lr=0.01, penalty_target=0., penalty_min=0., penalty_max=100.,
                div=None, loss='value', grad_pen=False, chi=False, regularize=False)
    args = NS(method=method, penalty=2., cliptarget=False, usereal=False,
              value_ratio=1., left=-0.1, right=0.1, log_interval=1000,
              only_expert_states=False, train=NS(use_target=True),
              q_net=NS(_target_='DoubleQCritic'), gamma=0.9)
    ens = DynamicsEnsemble(2, 1, N=len(offsets), effective_obs_dim=1, hidden_depth=0)
    ens.members = nn.ModuleList([ConstantDynamics(x) for x in offsets])
    actor = DiagGaussianActor(2, 1, 8, 1, [-1., 1.])
    critic = Critic()
    agent = NS(args=args, gamma=0.9, dynamics_ensemble=ens, actor=actor,
               critic=critic, critic_target=copy.deepcopy(critic))
    agent.getV = lambda obs: critic(obs, torch.zeros_like(obs[:, :1]))
    agent.get_targetV = lambda obs: agent.critic_target(obs, torch.zeros_like(obs[:, :1]))
    agent.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.001)
    obs = torch.tensor([[0., 7.], [1., 8.], [2., 9.], [3., 10.]])
    batch = (obs, obs + 0.1, torch.zeros(4, 1), torch.zeros(4, 1),
             torch.zeros(4, 1), torch.tensor([[False], [False], [True], [True]]))
    return agent, batch


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)

    def test_identical_members_zero_despite_stochastic_actor(self):
        agent, batch = fixture()
        for _ in range(5):
            gamma = iq._compute_dynamics_penalty(agent, batch)
            torch.testing.assert_close(gamma, torch.zeros(4, 1), atol=1e-7, rtol=0)
            self.assertFalse(gamma.requires_grad)
        # Independent member noise does produce spurious disagreement.
        states = agent.dynamics_ensemble.sample_next_ensemble(*[batch[i] for i in (0, 2)], M=10)
        actions = agent.actor.sample(states.reshape(-1, 2))[0]
        q = agent.critic_target(states.reshape(-1, 2), actions).reshape(4, 3, 10)
        self.assertGreater(q.mean(2).std(1, unbiased=False).mean().item(), 0.01)

    def test_known_disagreement_and_terminal_mask(self):
        agent, batch = fixture((-1., 0., 1.))
        agent.critic_target.action_sensitive = False
        batch[4][0] = 1  # true terminal; remaining rows allow bootstrap, including timeouts
        expected = 0.1 * 0.9 * torch.tensor([-1., 0., 1.]).std(unbiased=False)
        gamma = iq._compute_dynamics_penalty(agent, batch)
        torch.testing.assert_close(gamma[0], torch.zeros(1))
        torch.testing.assert_close(gamma[1:], expected.expand(3, 1))

    def test_shared_noise_padding_and_resampling(self):
        agent, batch = fixture()
        noise = torch.randn(4, 10, 1)
        ens = agent.dynamics_ensemble
        states = ens.sample_next_ensemble(batch[0], batch[2], M=10, noise=noise)
        torch.testing.assert_close(states[:, 0], states[:, 1])
        torch.testing.assert_close(states[..., 1], batch[0][:, 1, None, None].expand(4, 3, 10))
        with patch.object(ens, 'sample_next_ensemble', wraps=ens.sample_next_ensemble) as sample:
            iq._compute_dynamics_penalty(agent, batch)
            iq._compute_dynamics_penalty(agent, batch)
            self.assertFalse(torch.equal(sample.call_args_list[0].kwargs['noise'],
                                         sample.call_args_list[1].kwargs['noise']))

    def test_actor_noise_preserves_transform_logprob_and_gradients(self):
        agent, batch = fixture()
        obs = batch[0]
        noise = torch.randn(4, 1)
        dist = agent.actor(obs)
        action, log_prob, mean = agent.actor.sample(obs, noise=noise)
        expected = torch.tanh(dist.loc + dist.scale * noise)
        torch.testing.assert_close(action, expected)
        pre_tanh = dist.loc + dist.scale * noise
        expected_log_prob = (dist.base_dist.log_prob(pre_tanh)
                             - dist.transforms[0].log_abs_det_jacobian(pre_tanh, expected))
        torch.testing.assert_close(log_prob, expected_log_prob.sum(-1, keepdim=True))
        torch.testing.assert_close(mean, dist.mean)
        (action.sum() + log_prob.sum()).backward()
        self.assertTrue(all(p.grad is not None for p in agent.actor.parameters()))
        self.assertFalse(torch.equal(agent.actor.sample(obs)[0], agent.actor.sample(obs)[0]))

    def test_single_member_and_single_target_head(self):
        agent, batch = fixture((0.,))
        agent.critic_target = lambda obs, action: obs[:, :1]
        torch.testing.assert_close(iq._compute_dynamics_penalty(agent, batch), torch.zeros(4, 1))

    def test_invalid_sampling_configuration(self):
        agent, batch = fixture()
        agent.args.method.penalty_M = 0
        with self.assertRaises(ValueError):
            iq._compute_dynamics_penalty(agent, batch)
        agent.args.method.penalty_M = 10
        agent.args.method.penalty_N = 7
        with self.assertRaises(ValueError):
            iq._compute_dynamics_penalty(agent, batch)

    def test_loss_is_pure_and_reports_statistics_without_logging(self):
        agent, batch = fixture((-1., 0., 1.))
        gamma, weight = iq.prepare_iq_step(agent, batch)
        q = agent.critic(batch[0], batch[2])
        loss, logs, stat = iq.iq_loss(agent, q, q, torch.zeros_like(q), batch,
                                    penalty_u=gamma, constraint_penalty=weight)
        self.assertEqual(logs, {})
        self.assertFalse(stat.requires_grad)
        self.assertFalse(hasattr(agent, 'penalty_optimizer'))
        loss.backward()
        self.assertIsNotNone(agent.critic.weight.grad)
        for module in (agent.actor, agent.critic_target, agent.dynamics_ensemble):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        iq.update_iq_penalty(agent, [stat, stat * 2])
        torch.testing.assert_close(agent.log_penalty.grad, -1.5 * stat)
        self.assertEqual(agent.penalty_optimizer.state[agent.log_penalty]['step'].item(), 1)
        self.assertEqual(weight.item(), 2.)

    def test_all_entry_points_single_update(self):
        # Execute the actual critic-step function without importing simulator dependencies.
        for filename in ('train_iq.py', 'train_iq_noisy_expert.py',
                         'train_iq_offline.py', 'train_iq_robosuite.py'):
            source = ast.parse((ROOT / filename).read_text())
            function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'iq_update_critic')
            for double in (True, False):
                for uncertainty in (True, False):
                    with self.subTest(file=filename, double=double, uncertainty=uncertainty):
                        agent, batch = fixture((-1., 0., 1.))
                        agent.args.method.uncertainty = uncertainty
                        agent.args.q_net._target_ = 'DoubleQ' if double else 'SingleQ'
                        loss_spy = Mock(wraps=iq.iq_loss)
                        namespace = dict(torch=torch, iq_loss=loss_spy,
                                         prepare_iq_step=iq.prepare_iq_step,
                                         update_iq_penalty=iq.update_iq_penalty,
                                         synthetic_iq_loss=iq.synthetic_iq_loss,
                                         get_concat_samples=get_concat_samples,
                                         average_dicts=average_dicts)
                        exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
                        policy = tuple(x[:2] for x in batch[:5])
                        expert = tuple(x[2:] for x in batch[:5])
                        with patch.object(iq, '_compute_dynamics_penalty', wraps=iq._compute_dynamics_penalty) as sample:
                            namespace['iq_update_critic'](agent, policy, expert, Mock(), 1)
                        self.assertEqual(sample.call_count, int(uncertainty))
                        self.assertEqual(loss_spy.call_count, 2 if double else 1)
                        if double:
                            a, b = loss_spy.call_args_list
                            self.assertIs(a.kwargs['penalty_u'], b.kwargs['penalty_u'])
                            self.assertIs(a.kwargs['constraint_penalty'], b.kwargs['constraint_penalty'])
                        self.assertEqual(agent.penalty_optimizer.state[agent.log_penalty]['step'].item(), 1)

    def test_diagnostic_restores_rng_configuration_and_modes(self):
        from utils.uncertainty_diagnostics import diagnose_uncertainty
        agent, batch = fixture()
        agent.actor.trunk[0].eval()
        modes = [m.training for m in agent.actor.modules()]
        rng = torch.random.get_rng_state().clone()
        report = diagnose_uncertainty(agent, batch, sample_counts=(10, 50), repeats=5)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(agent.args.method.penalty_M, 10)
        self.assertEqual(modes, [m.training for m in agent.actor.modules()])
        shared = [r for r in report['results'] if r['sampling'] == 'shared']
        self.assertTrue(all(r['gamma_mean'] < 1e-7 for r in shared))
        independent = [r for r in report['results'] if r['sampling'] == 'independent']
        self.assertTrue(all(r['gamma_mean'] > 0 for r in independent))

    def test_disabled_constraints_do_not_update_weight(self):
        agent, batch = fixture()
        agent.args.method.constrain = False
        iq.update_iq_penalty(agent, [None])
        self.assertFalse(hasattr(agent, 'penalty_optimizer'))


if __name__ == '__main__':
    unittest.main()
