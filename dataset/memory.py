from collections import deque
import numpy as np
import random
import torch

from wrappers.atari_wrapper import LazyFrames
from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self._array_cache = None
        self._gpu_cache = {}

    def add(self, experience) -> None:
        self.buffer.append(experience)
        if self._array_cache is not None:
            self._array_cache = None
            self._gpu_cache = {}

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()
        self._array_cache = None
        self._gpu_cache = {}

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        if len(data) > self.memory_size:
            self.memory_size = len(data)
            self.buffer = deque(self.buffer, maxlen=self.memory_size)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])
        self._build_array_cache()

    def _build_array_cache(self):
        if len(self.buffer) == 0:
            self._array_cache = None
            self._gpu_cache = {}
            return

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
            *self.buffer)

        if isinstance(batch_state[0], LazyFrames):
            batch_state = np.array(batch_state, dtype=np.float32) / 255.0
        else:
            batch_state = np.asarray(batch_state, dtype=np.float32)
        if isinstance(batch_next_state[0], LazyFrames):
            batch_next_state = np.array(batch_next_state, dtype=np.float32) / 255.0
        else:
            batch_next_state = np.asarray(batch_next_state, dtype=np.float32)

        batch_action = np.asarray(batch_action, dtype=np.float32)
        if batch_action.ndim == 1:
            batch_action = np.expand_dims(batch_action, axis=1)
        batch_reward = np.asarray(batch_reward, dtype=np.float32).reshape(-1, 1)
        batch_done = np.asarray(batch_done, dtype=np.float32).reshape(-1, 1)

        self._array_cache = (
            batch_state, batch_next_state, batch_action, batch_reward, batch_done)
        self._gpu_cache = {}

    def _get_cached_samples(self, batch_size, device):
        device = torch.device(device)
        device_key = str(device)
        if device_key not in self._gpu_cache:
            self._gpu_cache[device_key] = tuple(
                torch.as_tensor(array, dtype=torch.float, device=device)
                for array in self._array_cache)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = \
            self._gpu_cache[device_key]
        if batch_size > batch_state.shape[0]:
            batch_size = batch_state.shape[0]
        indexes = torch.randint(batch_state.shape[0], (batch_size,), device=device)

        return (batch_state[indexes], batch_next_state[indexes],
                batch_action[indexes], batch_reward[indexes], batch_done[indexes])

    def get_samples(self, batch_size, device):
        if self._array_cache is not None:
            return self._get_cached_samples(batch_size, device)

        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
            *batch)

        # Scale obs for atari. TODO: Use flags
        if isinstance(batch_state[0], LazyFrames):
            # Use lazyframes for improved memory storage (same as original DQN)
            batch_state = np.array(batch_state) / 255.0
        if isinstance(batch_next_state[0], LazyFrames):
            batch_next_state = np.array(batch_next_state) / 255.0
        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done
