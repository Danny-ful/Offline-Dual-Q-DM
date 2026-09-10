"""HDF5 data-contract and CPU training coverage without a simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
from hydra import compose, initialize_config_dir
import numpy as np
import torch

from agent.dynamics_ensemble import load_iq_dynamics
from dataset.robomimic_dataset import load_robomimic_hdf5
import train_dynamics_robosuite as training


ROOT = Path(__file__).resolve().parents[1]
KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']


def write_dataset(path, task='lift', object_dim=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    widths = [object_dim or (10 if task == 'lift' else 14), 3, 4, 2]
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({
            'env_name': 'Lift' if task == 'lift' else 'PickPlaceCan',
            'env_kwargs': {'robots': ['Panda'], 'control_freq': 20,
                           'controller_configs': {'type': 'OSC_POSE'}}})
        # Success after three transitions, a failed trajectory, success at the
        # first transition, and a timeout. IDs overlap between PH and MG files.
        for i, rewards in enumerate(([0, 0, 1, 1, 1], [0] * 5, [1] * 5, [0] * 5)):
            demo = data.create_group(f'demo_{i}')
            for j, (key, width) in enumerate(zip(KEYS, widths)):
                obs = np.arange(5 * width, dtype=np.float32).reshape(5, width) / 100 + j + i
                demo.create_dataset('obs/' + key, data=obs)
                demo.create_dataset('next_obs/' + key, data=obs + .1)
            demo.create_dataset('actions', data=np.full((5, 7), .1 * (i + 1), dtype=np.float32))
            demo.create_dataset('rewards', data=rewards)
            # Deliberately unreliable flags; demo groups define boundaries.
            demo.create_dataset('dones', data=np.zeros(5))


def config(root, task='lift', overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'conf')):
        return compose(config_name='config_dynamics_robosuite', overrides=[
            f'robosuite.task={task}', f'robosuite.dataset_root="{root / "robomimic"}"',
            f'dyn.output_dir="{root / "output"}"', 'device=cpu',
            'method.penalty_N=5', 'dyn.epochs=2', 'dyn.batch_size=7',
            'dyn.hidden_dim=8', 'dyn.hidden_depth=1', 'dyn.val_frac=0.25', *overrides])


def write_pair(root, task='lift'):
    paths = [root / 'robomimic' / task / 'ph/low_dim_v15.hdf5',
             root / 'robomimic' / task / 'mg/low_dim_sparse_v15.hdf5']
    for path in paths:
        write_dataset(path, task)
    return paths


class RobosuiteDynamicsTrainingTests(unittest.TestCase):
    def test_iq_and_dynamics_share_order_truncation_and_all_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert, _ = write_pair(root)
            cfg = config(root)
            obs, act, nxt, ids, sources, metadata = training._build_dataset(cfg)
            iq_obs, iq_nxt, iq_act, _, dones, keys = load_robomimic_hdf5(expert, selected_keys=KEYS)
            np.testing.assert_array_equal(obs[:14], iq_obs)
            np.testing.assert_array_equal(nxt[:14], iq_nxt)
            np.testing.assert_array_equal(act[:14], iq_act)
            np.testing.assert_array_equal(np.flatnonzero(dones), [2, 7, 8, 13])
            self.assertEqual(obs.shape, (28, 19))
            self.assertEqual(keys, KEYS)
            self.assertEqual(ids[:4].tolist(), ['expert:demo_0'] * 3 + ['expert:demo_1'])
            self.assertEqual(ids[14], 'supplement:demo_0')
            self.assertEqual(metadata['sources']['expert']['trajectories'], 4)
            self.assertEqual(metadata['sources']['supplement']['transitions'], 14)
            # Configured order wins over HDF5's lexicographic key order.
            reordered = load_robomimic_hdf5(expert, max_trajs=1, selected_keys=KEYS[::-1])
            self.assertEqual(len(reordered[0]), 3)
            np.testing.assert_array_equal(reordered[0][:, :2], obs[:3, -2:])
            train, val = training._split_trajectories(ids, sources, .25, 0)
            self.assertFalse(set(ids[train]) & set(ids[val]))
            self.assertEqual(set(sources[val]), {'expert', 'supplement'})

    def test_invalid_hdf5_and_incompatible_sources_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert, supplement = write_pair(root)
            for limit in (0, -1, 1.5, True):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    load_robomimic_hdf5(expert, max_trajs=limit)
            for problem in ('next_obs', 'nan', 'length', 'layout', 'controller', 'task', 'missing'):
                with self.subTest(problem=problem):
                    write_pair(root)
                    if problem == 'layout':
                        write_dataset(supplement, object_dim=11)
                    elif problem == 'missing':
                        supplement.unlink()
                    else:
                        with h5py.File(supplement, 'a') as f:
                            demo = f['data/demo_0']
                            if problem == 'next_obs':
                                del demo['next_obs/robot0_eef_pos']
                            elif problem == 'nan':
                                demo['actions'][0, 0] = np.nan
                            elif problem == 'length':
                                del demo['actions']
                                demo.create_dataset('actions', data=np.zeros((4, 7)))
                            else:
                                args = json.loads(f['data'].attrs['env_args'])
                                if problem == 'controller':
                                    args['env_kwargs']['controller_configs']['type'] = 'JOINT_POSITION'
                                elif problem == 'task':
                                    args['env_name'] = 'PickPlaceCan'
                                f['data'].attrs['env_args'] = json.dumps(args)
                    with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                        training._build_dataset(config(root))

    def test_controller_metadata_aliases_are_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert, supplement = write_pair(root)
            for path, damping_key, limits_key in (
                    (expert, 'damping', 'damping_limits'),
                    (supplement, 'damping_ratio', 'damping_ratio_limits')):
                with h5py.File(path, 'a') as f:
                    args = json.loads(f['data'].attrs['env_args'])
                    controller = args['env_kwargs']['controller_configs']
                    controller['body_parts'] = {'right': {
                        'type': 'OSC_POSE', damping_key: 1, limits_key: [0, 10]}}
                    f['data'].attrs['env_args'] = json.dumps(args)
            _, _, _, _, _, metadata = training._build_dataset(config(root))
            self.assertEqual(metadata['sources']['expert']['transitions'], 14)

            # A real value difference must still be rejected after key normalization.
            with h5py.File(supplement, 'a') as f:
                args = json.loads(f['data'].attrs['env_args'])
                args['env_kwargs']['controller_configs']['body_parts']['right']['damping_ratio'] = .5
                f['data'].attrs['env_args'] = json.dumps(args)
            with self.assertRaisesRegex(ValueError, 'controller_configs'):
                training._build_dataset(config(root))

    def test_cpu_training_artifacts_and_iq_loading_for_both_tasks(self):
        with tempfile.TemporaryDirectory(prefix='robosuite dynamics ') as directory:
            root = Path(directory)
            for task, obs_dim in (('lift', 19), ('can', 23)):
                with self.subTest(task=task):
                    write_pair(root, task)
                    cfg = config(root, task)
                    cfg.dyn.check_only = True
                    training.main.__wrapped__(cfg)
                    self.assertFalse((root / 'output' / task).exists())
                    cfg.dyn.check_only = False
                    training.main.__wrapped__(cfg)
                    model_dir = (root / 'output' / task /
                                 'ph_low_dim__mg_low_dim_sparse')
                    checkpoint = model_dir / 'ensemble_5.pt'
                    self.assertTrue(checkpoint.is_file())
                    self.assertEqual([path.name for path in model_dir.iterdir()], ['ensemble_5.pt'])
                    payload = torch.load(checkpoint, weights_only=True)
                    metadata = payload['training_metadata']
                    self.assertEqual(payload['cfg']['obs_dim'], obs_dim)
                    self.assertEqual(metadata['dataset']['obs_keys'], KEYS)
                    self.assertEqual(len(metadata['split']['val_trajectories']), 2)
                    self.assertFalse(set(metadata['split']['train_trajectories']) &
                                     set(metadata['split']['val_trajectories']))
                    self.assertEqual(metadata['validation']['status'], 'ok')
                    self.assertEqual(set(metadata['validation']['by_source']), {'expert', 'supplement'})
                    # Existing IQ loader must accept, freeze, and sample this checkpoint.
                    cfg.method.uncertainty = True
                    cfg.method.dynamics_ckpt = str(checkpoint)
                    agent = SimpleNamespace(args=cfg)
                    load_iq_dynamics(agent, obs_dim, 7)
                    self.assertFalse(agent.dynamics_ensemble.training)
                    self.assertTrue(all(not p.requires_grad for p in agent.dynamics_ensemble.parameters()))
                    samples = agent.dynamics_ensemble.sample_next_ensemble(
                        torch.zeros(3, obs_dim), torch.zeros(3, 7), M=2)
                    self.assertEqual(samples.shape, (3, 5, 2, obs_dim))
                    self.assertTrue(torch.isfinite(samples).all())
                    cfg.robosuite.obs_keys = KEYS[::-1]
                    with self.assertRaisesRegex(ValueError, 'observation order'):
                        load_iq_dynamics(agent, obs_dim, 7)
                    cfg.robosuite.obs_keys = KEYS
                    cfg.robosuite.task = 'can' if task == 'lift' else 'lift'
                    with self.assertRaisesRegex(ValueError, 'task'):
                        load_iq_dynamics(agent, obs_dim, 7)

    def test_server_script_checks_all_inputs_and_forwards_overrides(self):
        with tempfile.TemporaryDirectory(prefix='robosuite server ') as directory:
            root = Path(directory)
            write_pair(root, 'lift')
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), CONDA_ENV='', PYTHON_BIN=sys.executable,
                       TASKS='lift can')
            args = ['bash', str(ROOT / 'scripts/run_dynamic_robosuite.sh'),
                    f'robosuite.dataset_root="{root / "robomimic"}"',
                    f'dyn.output_dir="{root / "output"}"', 'device=cpu',
                    'dyn.epochs=1', 'dyn.hidden_dim=8', 'dyn.hidden_depth=1',
                    'method.penalty_N=5']
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('can/ph/low_dim_v15.hdf5', result.stderr)
            self.assertFalse((root / 'output').exists())
            write_pair(root, 'can')
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoints = list((root / 'output').rglob('ensemble_5.pt'))
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len(list((root / 'output').rglob('*.*'))), 2)
            self.assertIn('Finished dynamics training for: lift can', result.stdout)


if __name__ == '__main__':
    unittest.main()
