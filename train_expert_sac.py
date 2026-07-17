"""
Train online SAC experts for MuJoCo environments (Ant, Hopper, HalfCheetah, Walker).
For each environment:
  1. Train SAC for 1M steps
  2. Save the entire replay buffer (1M steps) as supplement_buffer
  3. Collect 1 trajectory from the expert policy and save as expert_sac
"""
import datetime
import os
import pickle
import random
import time
from collections import deque
from itertools import count

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import evaluate, eval_mode


torch.set_num_threads(2)


class NoOpLogger:
    """Lightweight logger that discards all logs (avoids None crashes in SAC.update)."""
    def log(self, *args, **kwargs):
        pass
    def dump(self, *args, **kwargs):
        pass


# ENVS = ["HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
ENVS = ["Hopper-v2"]
TOTAL_STEPS = int(1e6)
INITIAL_MEMORY = 1280
EPISODE_STEPS = 1000
EVAL_INTERVAL = 10000
EVAL_EPISODES = 10
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "experts")
SUPPLEMENT_DIR = os.path.join(PROJECT_ROOT, "supplement")


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    return cfg


def collect_expert_trajectory(agent, env, max_steps=1000):
    """Collect one trajectory using the trained expert policy."""
    states, next_states, actions, rewards, dones = [], [], [], [], []
    state = env.reset()
    episode_reward = 0

    for _ in range(max_steps):
        with eval_mode(agent):
            action = agent.choose_action(state, sample=False)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        episode_reward += reward
        state = next_state
        if done:
            break

    traj = {
        "states": [tuple(states)],
        "next_states": [tuple(next_states)],
        "actions": [tuple(actions)],
        "rewards": [tuple(rewards)],
        "dones": [tuple(dones)],
        "lengths": [len(states)],
    }
    return traj, episode_reward


def train_single_env(args, env_name):
    """Train SAC on a single environment and save buffer + expert trajectory."""
    print(f"\n{'='*60}")
    print(f"Training SAC expert for: {env_name}")
    print(f"{'='*60}")

    args.env.name = env_name
    args.agent.name = "sac"
    args.agent.learn_temp = True
    args.num_seed_steps = 1000

    env = make_env(args)
    eval_env = make_env(args)
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)

    agent = make_agent(env, args)
    logger = NoOpLogger()

    replay_buffer = Memory(TOTAL_STEPS, args.seed)

    steps = 0
    learn_steps = 0
    begin_learn = False
    best_eval_returns = -np.inf
    rewards_window = deque(maxlen=100)
    eval_steps_log = []
    eval_returns_log = []

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False

        for episode_step in range(EPISODE_STEPS):
            if steps < args.num_seed_steps:
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            done_no_lim = done
            if (str(env.__class__.__name__).find('TimeLimit') >= 0
                    and episode_step + 1 == env._max_episode_steps):
                done_no_lim = 0
            replay_buffer.add((state, next_state, action, reward, done_no_lim))

            if replay_buffer.size() > INITIAL_MEMORY:
                if not begin_learn:
                    print(f"[{env_name}] Learning begins at step {steps}")
                    begin_learn = True

                learn_steps += 1
                if learn_steps >= TOTAL_STEPS:
                    print(f"[{env_name}] Finished {TOTAL_STEPS} learn steps!")
                    break

                agent.update(replay_buffer, logger, learn_steps)

                if learn_steps % EVAL_INTERVAL == 0:
                    eval_returns, _ = evaluate(agent, eval_env, num_episodes=EVAL_EPISODES)
                    mean_return = np.mean(eval_returns)
                    eval_steps_log.append(learn_steps)
                    eval_returns_log.append(mean_return)
                    print(f"[{env_name}] Step {learn_steps}/{TOTAL_STEPS} | "
                          f"Eval reward: {mean_return:.2f} | Best: {best_eval_returns:.2f}")
                    if mean_return > best_eval_returns:
                        best_eval_returns = mean_return

            if done:
                break
            state = next_state

        if learn_steps >= TOTAL_STEPS:
            break

        rewards_window.append(episode_reward)
        if epoch % 50 == 0:
            print(f"[{env_name}] Epoch {epoch} | Steps {learn_steps} | "
                  f"Train reward (avg): {np.mean(rewards_window):.2f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SUPPLEMENT_DIR, exist_ok=True)

    # Save supplement_buffer as trajectory-dict pkl (compatible with ExpertDataset/Memory.load)
    # buffer_tuples = list(replay_buffer.buffer)
    # states, next_states, actions, rewards, dones = zip(*buffer_tuples)
    # supplement_data = {
    #     "states": [np.array(states)],
    #     "next_states": [np.array(next_states)],
    #     "actions": [np.array(actions)],
    #     "rewards": [np.array(rewards)],
    #     "dones": [np.array(dones)],
    #     "lengths": [len(buffer_tuples)],
    # }
    # buffer_path = os.path.join(SUPPLEMENT_DIR, f"{env_name}_expert_sac.pkl")
    # with open(buffer_path, 'wb') as f:
    #     pickle.dump(supplement_data, f)
    # print(f"[{env_name}] Saved supplement_buffer ({replay_buffer.size()} steps) -> {buffer_path}")

    # Save model
    # model_path = os.path.join(SAVE_DIR, f"sac_{env_name}")
    # agent.save(model_path)
    # print(f"[{env_name}] Saved model -> {model_path}")

    # Collect and save 20 expert trajectories
    NUM_EXPERT_TRAJS = 20
    all_trajs = {"states": [], "next_states": [], "actions": [], "rewards": [], "dones": [], "lengths": []}
    traj_rewards = []
    for i in range(NUM_EXPERT_TRAJS):
        traj, traj_reward = collect_expert_trajectory(agent, eval_env, max_steps=EPISODE_STEPS)
        for key in all_trajs:
            all_trajs[key].extend(traj[key])
        traj_rewards.append(traj_reward)
        print(f"[{env_name}] Trajectory {i+1}/{NUM_EXPERT_TRAJS} reward: {traj_reward:.2f}, length: {traj['lengths'][0]}")
    expert_path = os.path.join(SAVE_DIR, f"{env_name}_expert_sac.pkl")
    with open(expert_path, 'wb') as f:
        pickle.dump(all_trajs, f)
    print(f"[{env_name}] Saved {NUM_EXPERT_TRAJS} expert trajectories "
          f"(avg reward: {np.mean(traj_rewards):.2f}) -> {expert_path}")

    # Plot training curve and save to project root
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_steps_log, eval_returns_log)
    ax.set_xlabel("Learn Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title(f"SAC Training - {env_name}")
    ax.grid(True)
    fig_path = os.path.join(PROJECT_ROOT, f"sac_training_{env_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[{env_name}] Saved training curve -> {fig_path}")

    env.close()
    eval_env.close()
    return best_eval_returns


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    results = {}
    for env_name in ENVS:
        best_return = train_single_env(args, env_name)
        results[env_name] = best_return

    print(f"\n{'='*60}")
    print("All training complete! Results:")
    print(f"{'='*60}")
    for env_name, ret in results.items():
        print(f"  {env_name}: best eval return = {ret:.2f}")


if __name__ == "__main__":
    main()
