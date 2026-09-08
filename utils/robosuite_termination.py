"""Success-as-terminal rules for standard single-arm Robosuite 1.5 observations.

Rules and object sensor order follow robosuite v1.5.2's lift.py and pick_place.py.
These operate on predicted observations; they do not reconstruct time limits or
failed-demonstration boundaries, and require unnormalized observation values.
"""
import torch


class RobosuiteTermination:
    def __init__(self, env_name, obs_shapes, env_kwargs=None):
        if env_name not in ('Lift', 'PickPlaceCan'):
            raise ValueError(f'No synthetic termination rule for Robosuite {env_name!r}')
        self.env_name = env_name
        kwargs = env_kwargs or {}
        offsets = {}
        width = 0
        for key, shape in obs_shapes.items():
            if len(shape) != 1:
                raise ValueError('Synthetic Robosuite constraints require flat low-dimensional observations')
            offsets[key] = width
            width += shape[0]
        self.obs_dim = width
        expected_object_dim = 10 if env_name == 'Lift' else 14
        if tuple(obs_shapes.get('object', ())) != (expected_object_dim,):
            raise ValueError(f'{env_name} requires the standard single-arm object observation '
                             f'({expected_object_dim} dimensions)')
        # Lift: cube position, quaternion, relative position.
        # Can: relative position, relative quaternion, world position, quaternion.
        self.object_pos = offsets['object'] + (0 if env_name == 'Lift' else 7)
        if env_name == 'PickPlaceCan':
            if tuple(obs_shapes.get('robot0_eef_pos', ())) != (3,):
                raise ValueError('PickPlaceCan requires robot0_eef_pos to check release distance')
            self.eef_pos = offsets['robot0_eef_pos']
            self.bin_pos = tuple(kwargs.get('bin2_pos', (0.1, 0.28, 0.8)))
            self.bin_size = tuple(kwargs.get('table_full_size', (0.39, 0.49, 0.82)))
            if len(self.bin_pos) != 3 or len(self.bin_size) != 3:
                raise ValueError('Robosuite bin2_pos and table_full_size must have three coordinates')

    @torch.no_grad()
    def __call__(self, next_obs):
        if next_obs.shape[-1] != self.obs_dim:
            raise ValueError('Synthetic observation width does not match the Robomimic layout')
        pos = next_obs[..., self.object_pos:self.object_pos + 3]
        if self.env_name == 'Lift':
            # Lift fixes the table height to 0.8 m and requires a 0.04 m margin.
            success = pos[..., 2:3] > 0.84
        else:
            # Can has object/bin id 3: upper x/y quadrant of the target bin.
            lower = next_obs.new_tensor(self.bin_pos)
            upper = next_obs.new_tensor((self.bin_pos[0] + self.bin_size[0] / 2,
                                         self.bin_pos[1] + self.bin_size[1] / 2,
                                         self.bin_pos[2] + 0.1))
            in_bin = ((pos > lower) & (pos < upper)).all(dim=-1, keepdim=True)
            eef = next_obs[..., self.eef_pos:self.eef_pos + 3]
            distance = torch.linalg.vector_norm(eef - pos, dim=-1, keepdim=True)
            released = (1 - torch.tanh(10 * distance)) < 0.6
            success = in_bin & released
        return success | ~torch.isfinite(next_obs).all(dim=-1, keepdim=True)
