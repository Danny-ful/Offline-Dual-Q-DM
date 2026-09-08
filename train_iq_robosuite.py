"""
Train IQ-Learn on Robomimic datasets and evaluate on Robosuite environments.

Usage:
    python train_iq_robosuite.py task=lift dataset_type=ph
"""

import datetime
import json
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
import h5py

from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss, prepare_iq_step, update_iq_penalty
from tqdm import tqdm

torch.set_num_threads(2)


def load_robomimic_env_metadata(hdf5_path):
    """Return the environment name and kwargs stored in a Robomimic dataset."""
    with h5py.File(hdf5_path, 'r') as f:
        env_args = f['data'].attrs.get('env_args')

    if env_args is None:
        return None, {}
    if isinstance(env_args, bytes):
        env_args = env_args.decode('utf-8')

    parsed_env_args = json.loads(env_args)
    env_kwargs = parsed_env_args.get('env_kwargs', {})
    if not isinstance(env_kwargs, dict):
        raise ValueError(f"Invalid env_kwargs in dataset metadata: {hdf5_path}")
    return parsed_env_args.get('env_name'), dict(env_kwargs)


def load_robomimic_hdf5(hdf5_path, max_trajs=None, selected_keys=None):
    """
    Load trajectories from robomimic HDF5 file.

    Returns:
        states, next_states, actions, rewards, dones, obs_keys (all numpy arrays + list of keys)
    """
    print(f"Loading data from: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Get all demo keys
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo')])

        if max_trajs is not None:
            demo_keys = demo_keys[:max_trajs]

        print(f"Found {len(demo_keys)} trajectories")
        if not demo_keys:
            raise ValueError(f"No demo trajectories found in {hdf5_path}")

        states_list = []
        next_states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        obs_keys = None

        for demo_key in tqdm(demo_keys, desc="Loading trajectories"):
            demo = f[f'data/{demo_key}']

            # Extract observations using the configured order.
            if obs_keys is None:
                obs_keys = (
                    list(selected_keys)
                    if selected_keys is not None
                    else sorted(demo['obs'].keys())
                )
                print(f"Dataset observation keys: {obs_keys}")

            missing = [key for key in obs_keys if key not in demo['obs']]
            if missing:
                raise KeyError(f"Missing observation keys: {missing}")

            if 'next_obs' not in demo:
                raise KeyError("Missing next_obs group")
            missing_next_obs = [key for key in obs_keys if key not in demo['next_obs']]
            if missing_next_obs:
                raise KeyError(f"Missing next observation keys: {missing_next_obs}")

            obs = np.concatenate(
                [demo['obs'][key][:] for key in obs_keys], axis=-1
            )  # (T, obs_dim)

            # Extract actions
            actions = demo['actions'][:]  # (T, action_dim)

            # Compute rewards (1 for success, 0 otherwise)
            # Robomimic datasets don't have explicit rewards, so we use success signal
            if 'rewards' in demo:
                rewards = np.asarray(demo['rewards'][:], dtype=np.float32)
            else:
                # Default: sparse reward at the end
                rewards = np.zeros(len(actions))
                if 'success' in demo.attrs and demo.attrs['success']:
                    rewards[-1] = 1.0

            # Create next_states
            next_obs = np.concatenate(
                [demo['next_obs'][key][:] for key in obs_keys], axis=-1
            )

            if len(rewards) != len(actions) or len(next_obs) != len(actions):
                raise ValueError(
                    f"Mismatched transition lengths in {demo_key}: "
                    f"actions={len(actions)}, rewards={len(rewards)}, "
                    f"next_obs={len(next_obs)}"
                )

            # The HDF5 dones field is a success signal rather than a reliable
            # episode boundary. A reward of 1 marks success; keep only the first
            # successful transition and terminate failed demos at their last step.
            success_steps = np.flatnonzero(np.isclose(rewards, 1.0))
            if len(success_steps):
                end = int(success_steps[0]) + 1
                obs = obs[:end]
                next_obs = next_obs[:end]
                actions = actions[:end]
                rewards = rewards[:end]

            dones = np.zeros(len(actions), dtype=np.float32)
            dones[-1] = 1.0

            states_list.append(obs)
            next_states_list.append(next_obs)
            actions_list.append(actions)
            rewards_list.append(rewards)
            dones_list.append(dones)

    # Concatenate all trajectories
    states = np.concatenate(states_list, axis=0)
    next_states = np.concatenate(next_states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    dones = np.concatenate(dones_list, axis=0)

    print(f"Loaded {len(states)} transitions")
    print(f"  State dim: {states.shape[1]}")
    print(f"  Action dim: {actions.shape[1]}")

    return states, next_states, actions, rewards, dones, obs_keys


class RobomimicMemory(object):
    """Memory buffer for robomimic datasets."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.memory_size = -1
        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.dones = None

    def size(self):
        return self.memory_size

    def load_hdf5(self, hdf5_path, max_trajs=None, selected_keys=None):
        """Load from robomimic HDF5 file."""
        states, next_states, actions, rewards, dones, obs_keys = load_robomimic_hdf5(
            hdf5_path, max_trajs, selected_keys
        )

        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.memory_size = self.states.shape[0]

        return obs_keys  # Return the observation keys

    def get_samples(self, batch_size, device):
        """Sample a batch of transitions."""
        idx = np.random.choice(np.arange(self.memory_size), size=batch_size, replace=True)

        batch_state = torch.as_tensor(self.states[idx], dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(self.next_states[idx], dtype=torch.float, device=device)
        batch_action = torch.as_tensor(self.actions[idx], dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(self.rewards[idx], dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(self.dones[idx], dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done


class FlattenObservationWrapper:
    """Wrapper to flatten dict observations to match dataset format."""
    def __init__(self, env, keys):
        self.env = env
        self.keys = keys

    def reset(self):
        obs_dict = self.env.reset()
        return self._flatten_obs(obs_dict)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs_dict), reward, done, info

    def _flatten_obs(self, obs_dict):
        """Flatten observation dict to vector matching dataset format."""
        import numpy as np
        obs_list = []
        for key in self.keys:
            if key in obs_dict:
                obs_list.append(np.array(obs_dict[key]).flatten())
            elif key == "object" and "object-state" in obs_dict:
                # "object" is mapped to "object-state" in newer versions
                obs_list.append(np.array(obs_dict["object-state"]).flatten())
            else:
                # Key not found in environment - this shouldn't happen if keys match dataset
                print(f"[WARNING] Key '{key}' from dataset not found in environment")
                print(f"Available env keys: {list(obs_dict.keys())}")
                raise KeyError(f"Dataset key '{key}' not available in environment")
        return np.concatenate(obs_list)

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_robosuite_env(
    task_name,
    robot=None,
    use_camera_obs=None,
    obs_keys=None,
    env_name=None,
    env_kwargs=None,
    horizon=None,
):
    """Create a robosuite environment.

    Args:
        task_name: Task name ('lift', 'can', etc.)
        robot: Robot type
        use_camera_obs: Whether to use camera observations
        obs_keys: List of observation keys to extract (if None, uses all available)
        env_name: Optional Robosuite environment name from dataset metadata
        env_kwargs: Optional Robosuite kwargs from dataset metadata
    """
    import robosuite as suite

    # Robosuite requires capitalized task names
    task_name_map = {
        'lift': 'Lift',
        'can': 'PickPlaceCan',
        'square': 'NutAssemblySquare',
        'round': 'NutAssemblyRound'
    }
    env_name = env_name or task_name_map.get(task_name.lower(), task_name.capitalize())
    suite_kwargs = dict(env_kwargs or {})

    suite_kwargs['env_name'] = env_name
    suite_kwargs.setdefault('has_renderer', False)
    suite_kwargs.setdefault('has_offscreen_renderer', False)
    suite_kwargs.setdefault('use_camera_obs', False)
    suite_kwargs.setdefault('robots', 'Panda')

    if robot is not None:
        suite_kwargs['robots'] = robot
    if use_camera_obs is not None:
        suite_kwargs['use_camera_obs'] = use_camera_obs
    if horizon is not None:
        suite_kwargs['horizon'] = int(horizon)

    env = suite.make(**suite_kwargs)

    # Wrap to flatten observations matching dataset
    if obs_keys is None:
        # If no keys provided, extract all available observations
        obs_dict_sample = env.reset()
        obs_keys = sorted(list(obs_dict_sample.keys()))
        print(f"[make_robosuite_env] Using all available keys: {obs_keys}")
    else:
        print(f"[make_robosuite_env] Using dataset keys: {obs_keys}")

    env = FlattenObservationWrapper(env, obs_keys)

    return env


def evaluate_robosuite(
    agent, env, num_episodes=10, stochastic=False, max_episode_steps=None
):
    """Evaluate agent on robosuite environment."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    episode_rewards = []
    episode_successes = []

    if max_episode_steps is None:
        max_episode_steps = env.horizon
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")

    for _ in range(num_episodes):
        obs = env.reset()

        episode_reward = 0
        episode_success = False
        done = False
        step = 0

        while not done and step < max_episode_steps:
            with eval_mode(agent):
                if stochastic:
                    action = agent.choose_action(obs, sample=True)
                else:
                    action = agent.choose_action(obs, sample=False)

            next_obs, reward, done, info = env.step(action)
            if 'success' in info:
                step_success = info.get('success')
            else:
                step_success = env._check_success()
            episode_success = bool(episode_success or step_success)

            episode_reward += reward
            obs = next_obs
            step += 1
            if episode_success:
                done = True


        episode_rewards.append(episode_reward)
        episode_successes.append(bool(episode_success))

    return episode_rewards, episode_successes


def get_args(cfg: DictConfig):
    # Allow adding new keys dynamically
    OmegaConf.set_struct(cfg, False)
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    OmegaConf.set_struct(cfg, True)
    print(OmegaConf.to_yaml(cfg))
    return cfg


# IQ-Learn update functions (same as train_iq_offline.py)
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

    penalty_u, constraint_penalty = prepare_iq_step(agent, batch)
    constraint_means = []

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1, constraint_mean = iq_loss(
            agent, current_Q1, current_V, next_V, batch,
            penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)
        q2_loss, loss_dict2, constraint_mean = iq_loss(
            agent, current_Q2, current_V, next_V, batch,
            penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict, constraint_mean = iq_loss(
            agent, current_Q, current_V, next_V, batch,
            penalty_u=penalty_u,
            constraint_penalty=constraint_penalty)
        constraint_means.append(constraint_mean)

    logger.log('train/critic_loss', critic_loss, step)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
    self.critic_optimizer.step()
    update_iq_penalty(agent, constraint_means)
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        actor_alpha_losses = {}
        if not self.args.agent.vdice_actor:
            obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


@hydra.main(config_path="conf", config_name="config_robosuite", version_base=None)
def main(cfg: DictConfig):

    args = get_args(cfg)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Get dataset paths
    task_name = args.robosuite.task  # "Lift" or "PickPlaceCan"

    # Expert dataset configuration
    expert_dataset_type = args.robosuite.expert_dataset_type  # "ph", "mh", "mg"
    expert_hdf5_type = args.robosuite.expert_hdf5_type  # "low_dim", "low_dim_sparse", "low_dim_dense"

    # Supplement dataset configuration
    supplement_dataset_type = args.robosuite.supplement_dataset_type  # "ph", "mh", "mg"
    supplement_hdf5_type = args.robosuite.supplement_hdf5_type  # "low_dim", "low_dim_sparse", "low_dim_dense"

    expert_hdf5 = f"robomimic/{task_name.lower()}/{expert_dataset_type}/{expert_hdf5_type}_v15.hdf5"
    supplement_hdf5 = f"robomimic/{task_name.lower()}/{supplement_dataset_type}/{supplement_hdf5_type}_v15.hdf5"
    expert_hdf5_path = hydra.utils.to_absolute_path(expert_hdf5)
    supplement_hdf5_path = hydra.utils.to_absolute_path(supplement_hdf5)

    if not args.robosuite.use_supplement:
        raise ValueError("Robosuite IQ-Learn requires a distinct supplement dataset for non-expert samples.")
    if os.path.abspath(expert_hdf5_path) == os.path.abspath(supplement_hdf5_path):
        raise ValueError("Expert and supplement datasets must be different files.")

    eval_env_name, dataset_env_kwargs = load_robomimic_env_metadata(expert_hdf5_path)
    eval_env_kwargs = dict(dataset_env_kwargs)
    eval_env_kwargs['ignore_done'] = False

    # Load expert data
    print("\n=== Loading Expert Data ===")
    print(f"Expert dataset: {expert_dataset_type}/{expert_hdf5_type}")
    expert_memory = RobomimicMemory(state_dim=-1, action_dim=-1)
    selected_keys = list(args.robosuite.obs_keys)
    dataset_obs_keys = expert_memory.load_hdf5(
        expert_hdf5_path,
        max_trajs=args.robosuite.expert_trajs,
        selected_keys=selected_keys,
    )
    print(f'--> Expert memory size: {expert_memory.size()}')

    # Load supplement/policy data
    print("\n=== Loading Supplement Data ===")
    print(f"Supplement dataset: {supplement_dataset_type}/{supplement_hdf5_type}")
    online_memory = RobomimicMemory(state_dim=-1, action_dim=-1)

    online_memory.load_hdf5(
        supplement_hdf5_path,
        max_trajs=args.robosuite.supplement_trajs,
        selected_keys=dataset_obs_keys,
    )
    print(f"--> Supplement memory size: {online_memory.size()}")

    if expert_memory.states.shape[1] != online_memory.states.shape[1]:
        raise ValueError(
            "Expert and supplement state dimensions differ: "
            f"{expert_memory.states.shape[1]} != {online_memory.states.shape[1]}"
        )
    if expert_memory.next_states.shape[1] != online_memory.next_states.shape[1]:
        raise ValueError("Expert and supplement next-state dimensions differ.")
    if expert_memory.actions.shape[1] != online_memory.actions.shape[1]:
        raise ValueError(
            "Expert and supplement action dimensions differ: "
            f"{expert_memory.actions.shape[1]} != {online_memory.actions.shape[1]}"
        )

    print(f"\n=== Dataset observation keys (will be used for eval env) ===")
    print(f"Keys: {dataset_obs_keys}")

    # Determine obs and action dims from loaded data
    obs_dim = expert_memory.states.shape[1]
    action_dim = expert_memory.actions.shape[1]

    # Create a dummy agent config with correct dimensions
    args.agent.obs_dim = obs_dim
    args.agent.action_dim = action_dim

    print(f"\nEnvironment dimensions:")
    print(f"  Observation: {obs_dim}")
    print(f"  Action: {action_dim}")

    run_name = args.exp_name or (
        f"robosuite_{task_name}_{expert_dataset_type}_expert_"
        f"{supplement_dataset_type}_supp"
    )
    safe_run_name = run_name.replace('/', '_').replace('\\', '_') or "robosuite"
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_dir = os.path.join(args.log_dir, safe_run_name, f"{ts_str}_seed{args.seed}")
    os.makedirs(log_dir, exist_ok=False)
    OmegaConf.save(args, os.path.join(log_dir, "config.yaml"))
    print(f'\n--> Saving logs at: {log_dir}')

    wandb.init(
        project=args.project_name,
        name=run_name,
        config=OmegaConf.to_container(args, resolve=True),
        reinit=True,
        sync_tensorboard=False,
    )

    # Create agent
    print("\n=== Creating Agent ===")

    # Create a dummy env object for agent initialization
    class DummyEnv:
        def __init__(self, obs_dim, action_dim):
            from gym import spaces
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,))

    dummy_env = DummyEnv(obs_dim, action_dim)
    agent = make_agent(dummy_env, args)

    # Load pretrained model if specified
    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print(f"=> Loading pretrain '{args.pretrain}'")
            agent.load(pretrain_path)
        else:
            print(f"[Attention]: Did not find checkpoint {args.pretrain}")

    # Logger writes each metric to TensorBoard, W&B, and CSV at log_interval.
    writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(
        log_dir,
        log_frequency=args.log_interval,
        writer=writer,
        save_tb=True,
        agent=args.agent.name
    )

    # Reuse one evaluation environment configured from the expert dataset metadata.
    eval_env = None
    if args.robosuite.eval_on_env:
        eval_env = make_robosuite_env(
            task_name=task_name,
            robot=args.robosuite.robot,
            obs_keys=dataset_obs_keys,
            env_name=eval_env_name,
            env_kwargs=eval_env_kwargs,
            horizon=args.robosuite.eval_horizon,
        )
        eval_obs = eval_env.reset()
        if eval_obs.shape != (obs_dim,):
            eval_env.close()
            raise ValueError(
                "Evaluation observation dimension does not match the training dataset."
            )

    # Training loop
    LEARN_STEPS = int(args.robosuite.learn_steps)
    print(f"\n=== Starting Training ({LEARN_STEPS} steps) ===\n")

    for step in tqdm(range(LEARN_STEPS)):
        # Bind IQ-Learn update functions
        agent.iq_update = types.MethodType(iq_update, agent)
        agent.iq_update_critic = types.MethodType(iq_update_critic, agent)

        # Update agent
        losses = agent.iq_update(online_memory, expert_memory, logger, step)

        # Log and flush training metrics at the configured cadence.
        if step % args.log_interval == 0:
            logger.dump(step, ty='train')

        # Evaluation on robosuite environment
        if step % args.robosuite.eval_interval == 0 and args.robosuite.eval_on_env:
            print(f"\n[Step {step}] Running evaluation on Robosuite...")

            eval_returns, eval_successes = evaluate_robosuite(
                agent,
                eval_env,
                num_episodes=args.robosuite.eval_episodes,
                stochastic=args.eval.stochastic
            )

            mean_return = np.mean(eval_returns)
            mean_success = np.mean(eval_successes)

            print(f"  Eval Return: {mean_return:.2f}")
            print(f"  Success Rate: {mean_success*100:.1f}%")

            # Evaluation must be recorded at its own cadence, even when it does
            # not divide the training logging interval.
            logger.log('eval/episode_reward', mean_return, step, log_frequency=1)
            logger.log('eval/success_rate', mean_success, step, log_frequency=1)
            logger.dump(step, ty='eval')

        # Save checkpoint
        if step % args.robosuite.save_interval == 0 and step > 0:
            save_path = os.path.join(log_dir, f'agent_{step}.pt')
            agent.save(save_path)
            print(f"\n[Step {step}] Saved checkpoint to {save_path}")

    if eval_env is not None:
        print("\n=== Final Evaluation ===")
        eval_returns, eval_successes = evaluate_robosuite(
            agent,
            eval_env,
            num_episodes=args.robosuite.eval_episodes * 2,
            stochastic=False,
        )

        print(f"\nFinal Results:")
        print(f"  Mean Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
        print(f"  Success Rate: {np.mean(eval_successes)*100:.1f}%")
        print(f"  Min/Max Return: {np.min(eval_returns):.2f} / {np.max(eval_returns):.2f}")
        eval_env.close()

    # Save final model
    final_save_path = os.path.join(log_dir, 'agent_final.pt')
    agent.save(final_save_path)
    print(f"\nSaved final model to {final_save_path}")

    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
