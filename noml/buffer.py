"""Replay buffer with optional Mirror Learning augmentation."""

from __future__ import annotations

import numpy as np
import torch

from .mirror import MirrorConfig, mirror_action, mirror_obs


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 500_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, act_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def _add_single(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
        mirror_cfg: MirrorConfig | None = None,
    ) -> None:
        """Add a transition. If mirror_cfg is provided, also add the mirrored twin."""
        self._add_single(obs, action, reward, next_obs, done)
        if mirror_cfg is not None and (mirror_cfg.obs_negate or mirror_cfg.act_negate):
            self._add_single(
                mirror_obs(obs, mirror_cfg),
                mirror_action(action, mirror_cfg),
                reward,
                mirror_obs(next_obs, mirror_cfg),
                done,
            )

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idx]),
            "action": torch.FloatTensor(self.action[idx]),
            "reward": torch.FloatTensor(self.reward[idx]),
            "next_obs": torch.FloatTensor(self.next_obs[idx]),
            "done": torch.FloatTensor(self.done[idx]),
        }

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            obs=self.obs[: self.size],
            action=self.action[: self.size],
            reward=self.reward[: self.size],
            next_obs=self.next_obs[: self.size],
            done=self.done[: self.size],
            ptr=np.array([self.ptr]),
            size=np.array([self.size]),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        n = int(data["size"][0])
        self.obs[:n] = data["obs"]
        self.action[:n] = data["action"]
        self.reward[:n] = data["reward"]
        self.next_obs[:n] = data["next_obs"]
        self.done[:n] = data["done"]
        self.ptr = int(data["ptr"][0])
        self.size = n
