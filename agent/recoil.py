"""ReCOIL agent (offline chi^2 dual objective with separate value network).

The agent bundles:
  - Q_phi: DoubleQCritic (with a target network for the Bellman residual target)
  - V_theta: a scalar state-value MLP (distinct from SAC's entropy-regularized V)
  - pi_psi: DiagGaussianActor, trained via advantage-weighted regression

The public interface (choose_action / save / load / critic / critic_target / gamma /
batch_size / device / args) mirrors agent.sac.SAC so that utils.evaluate and the
training loop in train_iq.py can reuse the same code paths.
"""
import numpy as np
import torch
from torch.optim import Adam
import hydra

import utils.utils as utils


class ReCOIL(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        # Q_phi and its target (reuse the SAC double-Q architecture)
        self.critic = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, _recursive_=False
        ).to(self.device)
        self.critic_target = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, _recursive_=False
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # pi_psi
        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        # V_theta: dedicated state-value head (not derived from Q + entropy)
        method_cfg = args.method
        self.value_net = utils.mlp(
            obs_dim,
            int(method_cfg.value_hidden_dim),
            1,
            int(method_cfg.value_hidden_depth),
        ).to(self.device)

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=agent_cfg.actor_lr,
            betas=agent_cfg.actor_betas,
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=agent_cfg.critic_lr,
            betas=agent_cfg.critic_betas,
        )
        self.value_optimizer = Adam(
            self.value_net.parameters(),
            lr=float(method_cfg.value_lr),
        )

        # Mirror SAC's optional linear scheduler so bc/iq hooks that reach into
        # self.scheduler keep working if anyone composes ReCOIL with them.
        self.she = args.schedular
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optimizer,
            start_factor=1.0,
            end_factor=(3e-6) / (5e-4),
            total_iters=300000,
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.value_net.train(training)

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        """State value via the dedicated V_theta head."""
        return self.value_net(obs)

    def get_targetV(self, obs):
        """Kept for interface parity; V_theta has no separate target network."""
        with torch.no_grad():
            return self.value_net(obs)

    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"
        value_path = f"{path}{suffix}_value"
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.value_net.state_dict(), value_path)

    def load(self, path, suffix=""):
        actor_path = f"{path}/{self.args.agent.name}{suffix}_actor"
        critic_path = f"{path}/{self.args.agent.name}{suffix}_critic"
        value_path = f"{path}/{self.args.agent.name}{suffix}_value"
        print(f"Loading ReCOIL models from {actor_path}, {critic_path}, {value_path}")
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        import os
        if os.path.isfile(value_path):
            self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.value_net(state).squeeze()
        return v.cpu().numpy()
