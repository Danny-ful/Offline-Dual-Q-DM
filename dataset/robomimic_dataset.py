"""Shared Robomimic HDF5 loading for IQ and offline dynamics training."""

import json

import h5py
import numpy as np
from tqdm import tqdm


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


def load_robomimic_hdf5(hdf5_path, max_trajs=None, selected_keys=None, *, return_metadata=False):
    """
    Load trajectories from robomimic HDF5 file.

    Returns:
        states, next_states, actions, rewards, dones, obs_keys.
        With return_metadata=True, append a dict containing per-transition demo
        IDs, observation shapes and retained trajectory lengths. The default
        six-value interface and first-success truncation are shared with IQ.
    """
    print(f"Loading data from: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Get all demo keys
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo')])

        if max_trajs is not None:
            if isinstance(max_trajs, bool) or not isinstance(max_trajs, int) or max_trajs < 1:
                raise ValueError("max_trajs must be a positive integer or null")
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
        obs_shapes = None
        action_shape = None
        trajectory_ids, trajectory_lengths = [], {}

        for demo_key in tqdm(demo_keys, desc="Loading trajectories"):
            demo = f[f'data/{demo_key}']

            # Extract observations using the configured order.
            if obs_keys is None:
                obs_keys = (
                    list(selected_keys)
                    if selected_keys is not None
                    else sorted(demo['obs'].keys())
                )
                if not obs_keys or len(set(obs_keys)) != len(obs_keys):
                    raise ValueError("Observation keys must be nonempty and unique")
                print(f"Dataset observation keys: {obs_keys}")

            missing = [key for key in obs_keys if key not in demo['obs']]
            if missing:
                raise KeyError(f"Missing observation keys: {missing}")

            if 'next_obs' not in demo:
                raise KeyError("Missing next_obs group")
            missing_next_obs = [key for key in obs_keys if key not in demo['next_obs']]
            if missing_next_obs:
                raise KeyError(f"Missing next observation keys: {missing_next_obs}")

            shapes = {key: list(demo['obs'][key].shape[1:]) for key in obs_keys}
            if any(len(shape) != 1 or shape[0] < 1 for shape in shapes.values()):
                raise ValueError(f"Expected flat low-dimensional observations in {demo_key}: {shapes}")
            if obs_shapes is not None and shapes != obs_shapes:
                raise ValueError(f"Observation layout changed in {demo_key}: {shapes} != {obs_shapes}")
            obs_shapes = shapes
            for key in obs_keys:
                if demo['next_obs'][key].shape != demo['obs'][key].shape:
                    raise ValueError(f"Observation/next_obs shape mismatch in {demo_key}/{key}")

            obs = np.concatenate(
                [demo['obs'][key][:] for key in obs_keys], axis=-1
            )  # (T, obs_dim)

            # Extract actions
            actions = demo['actions'][:]  # (T, action_dim)
            if actions.ndim != 2 or min(actions.shape) < 1:
                raise ValueError(f"Expected nonempty (T, action_dim) actions in {demo_key}")
            if action_shape is not None and actions.shape[1:] != action_shape:
                raise ValueError(f"Action dimension changed in {demo_key}")
            action_shape = actions.shape[1:]

            # Preserve stored rewards; use the existing IQ success fallback
            # only for archives without a rewards dataset.
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

            if (rewards.shape != (len(actions),) or len(obs) != len(actions)
                    or len(next_obs) != len(actions)):
                raise ValueError(
                    f"Mismatched transition lengths in {demo_key}: "
                    f"actions={len(actions)}, rewards={len(rewards)}, "
                    f"obs={len(obs)}, next_obs={len(next_obs)}"
                )
            if not all(np.isfinite(values).all() for values in (obs, next_obs, actions, rewards)):
                raise ValueError(f"Nonfinite data in {hdf5_path}:{demo_key}")

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
            trajectory_lengths[demo_key] = len(actions)
            if return_metadata:
                trajectory_ids.append(np.full(len(actions), demo_key))

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

    result = states, next_states, actions, rewards, dones, obs_keys
    if return_metadata:
        return (*result, {"trajectory_ids": np.concatenate(trajectory_ids),
                          "trajectory_lengths": trajectory_lengths,
                          "obs_shapes": obs_shapes})
    return result
