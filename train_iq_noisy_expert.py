"""
Train IQ-Learn with noisy expert data.
Expert data: 1 trajectory
Supplement data: constructed similarly to ISWBC noisy expert setup
- 10 trajectories with random actions
- 10 trajectories with original expert actions
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss, prepare_iq_step, update_iq_penalty, synthetic_iq_loss
from tqdm import tqdm
import pickle
from dataset.expert_dataset import ExpertDataset

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


class NoisyExpertMemory(object):
    """Memory buffer for noisy expert data constructed like ISWBC setup."""
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.memory_size = -1
        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.dones = None

    def size(self):
        return self.memory_size

    def load_noisy_expert(self, expert_path, num_expert_trajs, sample_freq, seed, action_dim):
        """
        Construct noisy expert dataset:
        - Load expert trajectories
        - Subsample: 10 trajs with random actions + 10 trajs with expert actions
        - Ensures no overlap between the two groups
        """
        # Load 20 trajectories at once to ensure no overlap
        dataset_all = ExpertDataset(expert_path, num_trajectories=20,
                                    subsample_frequency=sample_freq, seed=seed)

        total_len = len(dataset_all)
        split_point = total_len // 2

        # Collect data from first half (will have random actions)
        states1, next_states1, actions1, rewards1, dones1 = [], [], [], [], []
        for i in range(split_point):
            state, next_state, action, reward, done = dataset_all[i]
            states1.append(state)
            next_states1.append(next_state)
            actions1.append(action)
            rewards1.append(reward)
            dones1.append(done)

        # Collect data from second half (will keep expert actions)
        states2, next_states2, actions2, rewards2, dones2 = [], [], [], [], []
        for i in range(split_point, total_len):
            state, next_state, action, reward, done = dataset_all[i]
            states2.append(state)
            next_states2.append(next_state)
            actions2.append(action)
            rewards2.append(reward)
            dones2.append(done)

        # Convert to numpy arrays
        states1 = np.array(states1)
        actions1 = np.array(actions1)
        next_states1 = np.array(next_states1)
        rewards1 = np.array(rewards1)
        dones1 = np.array(dones1)

        states2 = np.array(states2)
        actions2 = np.array(actions2)
        next_states2 = np.array(next_states2)
        rewards2 = np.array(rewards2)
        dones2 = np.array(dones2)

        # Replace actions in first group with random actions
        random_actions = np.random.uniform(
            -1.0, 1.0,
            size=actions1.shape
        ).astype(actions1.dtype)

        # Concatenate to form final noisy dataset
        self.states = np.concatenate([states1, states2])
        self.next_states = np.concatenate([next_states1, next_states2])
        self.actions = np.concatenate([random_actions, actions2])
        self.rewards = np.concatenate([rewards1, rewards2])
        self.dones = np.concatenate([dones1, dones2])

        self.memory_size = self.states.shape[0]

        print(f"Noisy expert dataset constructed:")
        print(f"  - Random action trajs: 10 ({states1.shape[0]} transitions)")
        print(f"  - Expert action trajs: 10 ({states2.shape[0]} transitions)")
        print(f"  - Total: {self.memory_size} transitions")

    def get_samples(self, batch_size, device):
        idx = np.random.choice(np.arange(self.memory_size), size=batch_size, replace=False)

        batch_state = torch.as_tensor(self.states[idx], dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(self.next_states[idx], dtype=torch.float, device=device)
        batch_action = torch.as_tensor(self.actions[idx], dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(self.rewards[idx], dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(self.dones[idx], dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    args = get_args(cfg)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Fill dimensions before resolving OmegaConf interpolations used by q_net/actor configs.
    args.agent.obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        args.agent.action_dim = env.action_space.n
    else:
        args.agent.action_dim = env.action_space.shape[0]

    wandb.init(
        project=args.project_name,
        name=args.exp_name or None,
        config=OmegaConf.to_container(args, resolve=True),
        reinit=True,
        sync_tensorboard=True,
    )

    # Seed envs
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)

    if (getattr(args.method, "uncertainty", False)
            or getattr(args.method, "synthetic_constrain", False)):
        from agent.dynamics_ensemble import load_iq_dynamics
        load_iq_dynamics(agent, env.observation_space.shape[0], env.action_space.shape[0])

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Determine if we need to reduce observation dimension (for Ant-v2)
    reduce_obs_dim = None
    if args.env.name == 'Ant-v2' and args.env.get('reduce_obs_dim', False):
        reduce_obs_dim = args.env.get('effective_obs_dim', 27)
        print(f'--> Reducing observation dimension to {reduce_obs_dim} for {args.env.name}')

    # Load expert data
    expert_path = hydra.utils.to_absolute_path(args.env.expert_path)
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed, reduce_obs_dim=reduce_obs_dim)
    expert_memory_replay.load(expert_path,
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    # Load supplement data
    supplement_path = hydra.utils.to_absolute_path(args.env.supplement_path)
    if not os.path.isfile(supplement_path):
        raise FileNotFoundError(
            f"Supplement dataset not found at {supplement_path}."
        )
    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed + 1, reduce_obs_dim=reduce_obs_dim)
    online_memory_replay.load(supplement_path,
                              num_trajs=np.iinfo(np.int32).max,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 43)
    print(f"--> Supplement memory size: {online_memory_replay.size()}")

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, "noisy_expert")

    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')

    logger = Logger(log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)


    for step in tqdm(range(LEARN_STEPS)):
        agent.iq_update = types.MethodType(iq_update, agent)
        agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
        losses = agent.iq_update(online_memory_replay,
                                    expert_memory_replay, logger, step)
        if step % 1000 == 0:
            for key, loss in losses.items():
                writer.add_scalar(key, loss, global_step=step)

        if step % args.env.eval_interval == 0:
            eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps,
                                                    stochastic=args.eval.stochastic)
            returns = np.mean(eval_returns)
            logger.log('eval/episode_reward', returns, step)
            logger.dump(step, ty='eval')





# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch


    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch


    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    log_this_step = (step % 1000 == 0)

    penalty_u, constraint_penalty = prepare_iq_step(agent, batch)
    constraint_means = []

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1, constraint_mean = iq_loss(
            agent, current_Q1, current_V, next_V, batch,
            log_this_step=log_this_step, penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)
        q2_loss, loss_dict2, constraint_mean = iq_loss(
            agent, current_Q2, current_V, next_V, batch,
            log_this_step=log_this_step, penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
        if log_this_step:
            loss_dict['Q_mean'] = 0.5 * (current_Q1.mean().item() + current_Q2.mean().item())
            loss_dict['Q_max'] = max(current_Q1.max().item(), current_Q2.max().item())
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict, constraint_mean = iq_loss(
            agent, current_Q, current_V, next_V, batch,
            log_this_step=log_this_step, penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)
        if log_this_step:
            loss_dict['Q_mean'] = current_Q.mean().item()
            loss_dict['Q_max'] = current_Q.max().item()

    synthetic_loss, synthetic_logs = synthetic_iq_loss(
        agent, obs, step, log_this_step=step % args.log_interval == 0)
    critic_loss = critic_loss + synthetic_loss
    loss_dict.update(synthetic_logs)
    if 'total_loss' in loss_dict:
        loss_dict['total_loss'] = critic_loss.item()

    logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    update_iq_penalty(agent, constraint_means)
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            # if self.args.offline:
            #     obs = expert_batch[0]
            # else:
            #     # Use both policy and expert observations
            obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()