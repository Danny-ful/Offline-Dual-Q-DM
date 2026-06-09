"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
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
from agent.bc import bc_update
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss
from recoil import recoil_update

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    wandb.init(project=args.project_name,
               sync_tensorboard=False,
               reinit=True,
               config=OmegaConf.to_container(args, resolve=False))
    wandb.define_metric("learn_steps")
    wandb.define_metric("train/*", step_metric="learn_steps")
    wandb.define_metric("eval/*", step_metric="learn_steps")

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

    # Seed envs
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent = make_agent(env, args)

    if getattr(args.method, "penalty", False):
        from agent.dynamics_ensemble import DynamicsEnsemble
        ckpt_path = args.method.dynamics_ckpt
        if not ckpt_path:
            raise ValueError(
                "method.penalty=True but method.dynamics_ckpt is empty. "
                "Please first run train_dynamics.py to produce the checkpoint."
            )
        ckpt_abs = hydra.utils.to_absolute_path(ckpt_path)
        ens = DynamicsEnsemble(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            N=int(args.method.penalty_N),
        ).to(args.device)
        ens.load(ckpt_abs, map_location=args.device)
        ens.eval()
        for p in ens.parameters():
            p.requires_grad_(False)
        agent.dynamics_ensemble = ens
        print(f"--> Loaded dynamics ensemble (N={args.method.penalty_N}) from {ckpt_abs}")

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    is_bc = args.method.type == "bc"
    is_recoil = args.method.type == "recoil"
    if is_recoil and not bool(args.offline):
        raise ValueError(
            "ReCOIL is an offline algorithm; please run with offline=True "
            "(it needs a suboptimal dataset under supplement/)."
        )
    # BC trains purely from expert data; reuse the offline loop (no env interaction).
    data_only = bool(args.offline) or is_bc

    demo_filename = os.path.basename(args.env.demo)

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{demo_filename}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    if data_only and not is_bc:
        supplement_path = hydra.utils.to_absolute_path(f'supplement/{demo_filename}')
        if not os.path.isfile(supplement_path):
            raise FileNotFoundError(
                f"Offline mode requires supplement data at {supplement_path}. "
                "Please export the matching file into supplement/ first."
            )
        # Keep expert.demos for expert loading, but load the full matching
        # supplement dataset into the policy buffer.
        online_memory_replay.load(supplement_path,
                                  num_trajs=np.iinfo(np.int32).max,
                                  sample_freq=args.expert.subsample_freq,
                                  seed=args.seed + 43)
        # supplement_num_trajs = args.expert.demos*10
        # online_memory_replay.load(supplement_path,
        #                           num_trajs=supplement_num_trajs,
        #                           sample_freq=args.expert.subsample_freq,
        #                           seed=args.seed + 43)
        print(f'--> Supplement memory size: {online_memory_replay.size()}')

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    steps = 0

    # track mean reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf
    bc_early_stop = bool(is_bc and getattr(args.method, "early_stop", False))
    bc_early_stop_patience = int(getattr(args.method, "early_stop_patience", 20))
    bc_early_stop_min_delta = float(getattr(args.method, "early_stop_min_delta", 0.0))
    bc_no_improve_evals = 0

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    # Sample initial states from env
    state_0 = [env.reset()] * INITIAL_STATES
    if isinstance(state_0[0], LazyFrames):
        state_0 = np.array(state_0) / 255.0
    state_0 = torch.FloatTensor(np.array(state_0)).to(args.device)

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False

        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):
            if data_only:
                if learn_steps % args.env.eval_interval == 0:
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                    returns = np.mean(eval_returns)
                    learn_steps += 1  # To prevent repeated eval at timestep 0
                    logger.log('eval/episode_reward', returns, learn_steps)
                    logger.log('eval/episode_reward_std', np.std(eval_returns), learn_steps)
                    logger.log('eval/episode_reward_min', np.min(eval_returns), learn_steps)
                    logger.log('eval/episode_reward_max', np.max(eval_returns), learn_steps)
                    logger.log('eval/episode', epoch, learn_steps)

                    improved = returns > best_eval_returns
                    improved_for_early_stop = returns > best_eval_returns + bc_early_stop_min_delta

                    if improved:
                        # Store best eval returns
                        best_eval_returns = returns
                        wandb.run.summary["best_returns"] = best_eval_returns
                        save(agent, epoch, args, output_dir='results_best')

                    # if bc_early_stop:
                    #     if improved_for_early_stop:
                    #         bc_no_improve_evals = 0
                    #     else:
                    #         bc_no_improve_evals += 1

                    # if bc_early_stop:
                    #     logger.log('eval/bc_no_improve_evals', bc_no_improve_evals, learn_steps)
                    # logger.dump(learn_steps, ty='eval')

                    # if bc_early_stop and bc_no_improve_evals >= bc_early_stop_patience:
                    #     print(
                    #         "Early stopping BC after "
                    #         f"{bc_no_improve_evals} evals without improvement "
                    #         f"(best={best_eval_returns:.2f}, current={returns:.2f}, "
                    #         f"min_delta={bc_early_stop_min_delta:.2f})."
                    #     )
                    #     wandb.run.summary["early_stop_step"] = learn_steps
                    #     wandb.run.summary["early_stop_best_returns"] = best_eval_returns
                    #     wandb.finish()
                    #     return

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    wandb.finish()
                    return

                if is_bc:
                    # BC update using expert data only.
                    agent.bc_update = types.MethodType(bc_update, agent)
                    losses = agent.bc_update(expert_memory_replay, logger, learn_steps)
                elif is_recoil:
                    # ReCOIL: three-way update over expert + supplement mixture.
                    agent.recoil_update = types.MethodType(recoil_update, agent)
                    losses = agent.recoil_update(online_memory_replay,
                                                 expert_memory_replay, logger, learn_steps)
                else:
                    # IQ-Learn update without environment interaction.
                    agent.iq_update = types.MethodType(iq_update, agent)
                    agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                    losses = agent.iq_update(online_memory_replay,
                                             expert_memory_replay, logger, learn_steps)

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)
                        wandb.log({key: loss, "learn_steps": learn_steps}, step=learn_steps)
                continue

            if steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                logger.log('eval/episode_reward', returns, learn_steps)
                logger.log('eval/episode_reward_std', np.std(eval_returns), learn_steps)
                logger.log('eval/episode_reward_min', np.min(eval_returns), learn_steps)
                logger.log('eval/episode_reward_max', np.max(eval_returns), learn_steps)
                logger.log('eval/episode', epoch, learn_steps)
                logger.dump(learn_steps, ty='eval')
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = returns
                    wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, epoch, args, output_dir='results_best')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            if online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    wandb.finish()
                    return

                ######
                # IQ-Learn Modification
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                losses = agent.iq_update(online_memory_replay,
                                         expert_memory_replay, logger, learn_steps)
                ######

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)
                        wandb.log({key: loss, "learn_steps": learn_steps}, step=learn_steps)

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        logger.log('train/episode', epoch, learn_steps)
        logger.log('train/episode_reward', episode_reward, learn_steps)
        logger.log('train/duration', time.time() - start_time, learn_steps)
        logger.dump(learn_steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')


def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        elif args.method.type == "recoil":
            name = f'recoil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        # Isolate checkpoints per W&B run to avoid cross-agent collisions in sweeps.
        run_id = getattr(wandb.run, "id", None) if wandb.run is not None else None
        run_output_dir = os.path.join(output_dir, run_id or "offline")
        os.makedirs(run_output_dir, exist_ok=True)
        agent.save(f'{run_output_dir}/{args.agent.name}_{name}')


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

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)

        # Track Q1-Q2 disagreement: a temperature gauge for DoubleQ stability.
        # Healthy: rises and plateaus. Oscillating -> race-to-bottom on min target.
        # Collapses to 0 -> heads degenerate, DoubleQ loses its benefit.
        with torch.no_grad():
            q_disagree = (current_Q1 - current_Q2).abs().mean()
        loss_dict["diagnostics/q_disagree"] = q_disagree.item()
        logger.log("train/q_disagree", q_disagree, step)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    if getattr(self, "actor", None) is not None:
        with torch.no_grad():
            expert_actor_action = self.actor(expert_obs).mean
            policy_actor_action = self.actor(policy_obs).mean
            q_expert_action = self.critic(expert_obs, expert_action).mean()
            q_actor_action = self.critic(expert_obs, expert_actor_action).mean()
            q_gap = q_expert_action - q_actor_action
            q_policy_dataset_action = self.critic(policy_obs, policy_action).mean()
            q_policy_actor_action = self.critic(policy_obs, policy_actor_action).mean()
            q_policy_gap = q_policy_dataset_action - q_policy_actor_action
            expert_actor_action_mse = F.mse_loss(expert_actor_action, expert_action)
            sup_dataset_mse = F.mse_loss(policy_actor_action, policy_action)

        # loss_dict["diagnostics/q_expert_state_expert_action"] = q_expert_action.item()
        # loss_dict["diagnostics/q_expert_state_actor_mean_action"] = q_actor_action.item()
        loss_dict["diagnostics/q_expert_state_expert_minus_actor"] = q_gap.item()
        # loss_dict["diagnostics/q_supplement_state_dataset_action"] = q_policy_dataset_action.item()
        # loss_dict["diagnostics/q_supplement_state_actor_mean_action"] = q_policy_actor_action.item()
        loss_dict["diagnostics/q_supplement_state_dataset_minus_actor"] = q_policy_gap.item()
        loss_dict["diagnostics/actor_action_mse_to_expert_on_expert_obs"] = expert_actor_action_mse.item()
        loss_dict["diagnostics/actor_action_mse_to_dataset_on_supplement_obs"] = sup_dataset_mse.item()
        # logger.log("train/q_expert_state_expert_action", q_expert_action, step)
        # logger.log("train/q_expert_state_actor_mean_action", q_actor_action, step)
        logger.log("train/q_expert_state_expert_minus_actor", q_gap, step)
        # logger.log("train/q_supplement_state_dataset_action", q_policy_dataset_action, step)
        # logger.log("train/q_supplement_state_actor_mean_action", q_policy_actor_action, step)
        logger.log("train/q_supplement_state_dataset_minus_actor", q_policy_gap, step)
        logger.log("train/actor_action_mse_to_expert_on_expert_obs", expert_actor_action_mse, step)
        logger.log("train/actor_action_mse_to_dataset_on_supplement_obs", sup_dataset_mse, step)

    logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()

    # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=10.0)

    # step critic
    self.critic_optimizer.step()
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(4*self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline and self.args.actor_expert_offline:
                obs = expert_batch[0]
            else:
                # Use both policy/supplement and expert observations.
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    if self.args.actor_bc:
                        actor_alpha_losses = self.update_actor_and_alpha(
                            obs, logger, step, expert_batch[0], expert_batch[2])
                    else:
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
