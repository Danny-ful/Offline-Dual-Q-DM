"""Fit one offline dynamics ensemble per Robomimic task, without a simulator."""
from __future__ import annotations

import os
from pathlib import Path
import random
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from agent.dynamics_ensemble import DynamicsEnsemble
from dataset.robomimic_dataset import load_robomimic_env_metadata, load_robomimic_hdf5
from train_dynamics import _diagnostics, _split_trajectories, _train_ensemble


def _dataset_paths(cfg):
    robo = cfg.robosuite
    task = str(robo.task).lower()
    if task not in ('lift', 'can'):
        raise ValueError('robosuite.task must be lift or can')
    if not robo.use_supplement:
        raise ValueError('This trainer requires both expert and supplement datasets')
    root = Path(hydra.utils.to_absolute_path(str(robo.dataset_root)))
    paths = {
        source: root / task / str(robo[f'{source}_dataset_type']) / f"{robo[f'{source}_hdf5_type']}_v15.hdf5"
        for source in ('expert', 'supplement')
    }
    for source, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{source} dataset not found at {path}')
    if paths['expert'].resolve() == paths['supplement'].resolve():
        raise ValueError('Expert and supplement datasets must be different files')
    return paths


def _check_env_metadata(paths, task):
    metadata = {}
    expected = {'lift': 'Lift', 'can': 'PickPlaceCan'}[task]
    for source, path in paths.items():
        name, kwargs = load_robomimic_env_metadata(path)
        # Robomimic Can archives may name the generic PickPlace environment.
        allowed = {expected, 'PickPlace'} if task == 'can' else {expected}
        if name is not None and name not in allowed:
            raise ValueError(f'{source} environment {name!r} does not match task {task}')
        if name is None:
            warnings.warn(f'{source}: missing environment metadata; compatibility cannot be fully checked')
        metadata[source] = {'env_name': name, 'env_kwargs': kwargs}
    # Reward shaping, rendering and horizon settings need not agree. Robot,
    # controller and timing differences change the transition/action semantics.
    expert, supplement = (metadata[source]['env_kwargs'] for source in paths)
    for key in ('robots', 'controller_configs', 'gripper_types', 'control_freq',
                'base_types', 'single_object_mode', 'object_type'):
        if key in expert and key in supplement:
            left, right = expert[key], supplement[key]
            if key in ('robots', 'gripper_types', 'base_types'):
                left = left if isinstance(left, list) else [left]
                right = right if isinstance(right, list) else [right]
            if left != right:
                raise ValueError(f'Expert/supplement environment metadata mismatch for {key}: {left!r} != {right!r}')
    return metadata


def _build_dataset(cfg):
    paths = _dataset_paths(cfg)
    task = str(cfg.robosuite.task).lower()
    env_metadata = _check_env_metadata(paths, task)
    arrays = [[], [], []]
    ids, sources = [], []
    metadata = {'task': task, 'obs_keys': list(cfg.robosuite.obs_keys),
                'action_scaling': 'dataset_original',
                'trajectory_processing': 'truncate_after_first_reward_one', 'sources': {}}
    layout = None
    for source, path in paths.items():
        obs, nxt, actions, _, _, _, info = load_robomimic_hdf5(
            str(path), max_trajs=cfg.robosuite[f'{source}_trajs'],
            selected_keys=metadata['obs_keys'], return_metadata=True)
        current_layout = (info['obs_shapes'], actions.shape[1])
        if layout is not None and current_layout != layout:
            raise ValueError('Expert/supplement observation layout or action dimensions differ')
        layout = current_layout
        for target, values in zip(arrays, (obs, actions, nxt)):
            values = np.asarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'{source}: data is not finite in float32')
            target.append(values)
        ids.append(np.asarray([f'{source}:{demo}' for demo in info['trajectory_ids']]))
        sources.append(np.full(len(obs), source))
        metadata['sources'][source] = {
            'path': str(path), 'trajectories': len(info['trajectory_lengths']),
            'transitions': len(obs), 'trajectory_lengths': info['trajectory_lengths'],
            'environment': env_metadata[source],
            'action_min': actions.min(axis=0).tolist(), 'action_max': actions.max(axis=0).tolist(),
        }
    metadata['obs_shapes'], metadata['action_dim'] = layout
    obs, actions, nxt = (np.concatenate(values) for values in arrays)
    return obs, actions, nxt, np.concatenate(ids), np.concatenate(sources), metadata


@hydra.main(version_base=None, config_path='conf', config_name='config_dynamics_robosuite')
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == 'auto':
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(cfg.device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable; use device=cpu or device=auto')
    if cfg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dyn = cfg.dyn
    if int(cfg.method.penalty_N) != 5:
        raise ValueError('Robosuite dynamics output is fixed to a five-member ensemble; use method.penalty_N=5')
    if min(int(dyn.epochs), int(dyn.batch_size), int(dyn.log_interval),
           int(dyn.hidden_dim), int(dyn.diagnostic_bins)) < 1:
        raise ValueError('Training sizes and intervals must be positive')
    if int(dyn.hidden_depth) < 0:
        raise ValueError('dyn.hidden_depth must be nonnegative')
    for key in ('lr', 'normalization_eps', 'weight_decay'):
        value = float(dyn[key])
        if not np.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError(f'Invalid dyn.{key}: {value}')

    obs, actions, nxt, ids, sources, dataset_metadata = _build_dataset(cfg)
    obs_dim, action_dim = obs.shape[1], actions.shape[1]
    cfg.agent.obs_dim, cfg.agent.action_dim = obs_dim, action_dim
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    train_np, val_np = _split_trajectories(ids, sources, float(dyn.val_frac), cfg.seed)
    print(f'--> task={cfg.robosuite.task} obs_dim={obs_dim} action_dim={action_dim} | '
          f'train={len(train_np)} val={len(val_np)} transitions | '
          f'train={len(np.unique(ids[train_np]))} val={len(np.unique(ids[val_np]))} trajectories')
    if dyn.check_only:
        print('--> Dataset checks passed; no dynamics training or checkpoint output')
        return

    robo = cfg.robosuite
    combination = (f'{robo.expert_dataset_type}_{robo.expert_hdf5_type}__'
                   f'{robo.supplement_dataset_type}_{robo.supplement_hdf5_type}')
    save_dir = (Path(hydra.utils.to_absolute_path(str(dyn.output_dir))) /
                dataset_metadata['task'] / combination)
    save_dir.mkdir(parents=True, exist_ok=True)

    obs_t, act_t, next_t = (torch.as_tensor(values, dtype=torch.float32, device=cfg.device)
                          for values in (obs, actions, nxt))
    train_idx, val_idx = (torch.as_tensor(values, device=cfg.device) for values in (train_np, val_np))
    ensemble = DynamicsEnsemble(obs_dim, action_dim, N=int(cfg.method.penalty_N),
                                hidden_dim=int(dyn.hidden_dim), hidden_depth=int(dyn.hidden_depth),
                                effective_obs_dim=obs_dim).to(cfg.device)
    training = _train_ensemble(ensemble, obs_t, act_t, next_t, train_idx, val_idx, dyn, cfg.seed)
    diagnostics, _ = _diagnostics(ensemble, obs_t, act_t, next_t, val_idx, ids, sources,
                                  int(dyn.batch_size), int(dyn.diagnostic_bins))
    training.update(config=resolved_config, dataset=dataset_metadata,
                    validation=diagnostics,
                    split={'unit': 'trajectory', 'val_frac': float(dyn.val_frac),
                           'train_trajectories': np.unique(ids[train_np]).tolist(),
                           'val_trajectories': np.unique(ids[val_np]).tolist()})
    checkpoint = save_dir / 'ensemble_5.pt'
    temporary = save_dir / '.ensemble_5.pt.tmp'
    ensemble.save(str(temporary), training_metadata=training)
    os.replace(temporary, checkpoint)
    print(f'--> Saved {checkpoint} (replaces an existing checkpoint at this path)')
    print(f'--> IQ override: method.penalty_N={ensemble.N} method.dynamics_ckpt="{checkpoint}"')


if __name__ == '__main__':
    main()
