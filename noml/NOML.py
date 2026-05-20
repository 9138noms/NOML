"""NOML — Hierarchical TD3 + Anchor Policy + Mirror Learning.

Core formula:
    action = clip(anchor + delta * gate, -1, +1)   for continuous flight axes
    fire   = sigmoid(MLP(obs))                     for optional binary axis

Where:
    anchor    : fixed safe action (e.g., wings level, MIL throttle)
    delta     : hierarchical actor output, level i conditions on levels < i
    gate      : per-axis sigmoid, "how far to deviate from anchor"

See README.md for the full architecture description.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .mirror import MirrorConfig
from .networks import ActorLevel, GateNetwork, TwinCritic, build_fire_head


class NOML:
    """Hierarchical TD3 with anchor policy and mirror learning.

    The hierarchical actor is structured as three levels for the continuous
    flight axes (typical for fixed-wing aircraft):
        L0: obs                          -> d_pitch        (1 dim)
        L1: obs + d_pitch                -> d_roll         (1 dim)
        L2: obs + d_pitch + d_roll       -> remaining      (act_dim - 3 dims,
                                                            excludes optional
                                                            binary axis)

    If `has_fire_head` is True, the last action dimension is treated as an
    independent binary head (sigmoid, no anchor/gate), with epsilon-greedy
    exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        anchor: np.ndarray,
        mirror_cfg: MirrorConfig | None = None,
        has_fire_head: bool = False,
        hidden: list[int] | None = None,
        gate_hidden: list[int] | None = None,
        lr_actor: float = 3e-4,
        lr_gate: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.002,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        fire_explore_prob: float = 0.3,
        buffer_size: int = 500_000,
        batch_size: int = 256,
        learning_starts: int = 25_000,
    ):
        if hidden is None:
            hidden = [256, 256]
        if gate_hidden is None:
            gate_hidden = [128, 128]
        if anchor.shape != (act_dim,):
            raise ValueError(f"anchor shape {anchor.shape} != (act_dim={act_dim},)")

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.has_fire_head = has_fire_head
        self.flight_dim = act_dim - 1 if has_fire_head else act_dim

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.fire_explore_prob = fire_explore_prob
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.mirror_cfg = mirror_cfg

        self.anchor = torch.FloatTensor(anchor)

        # Hierarchical delta actors. L2 outputs (flight_dim - 2) values.
        rest_dim = self.flight_dim - 2
        if rest_dim < 0:
            raise ValueError(f"flight_dim must be >= 2, got {self.flight_dim}")
        self.actor_l0 = ActorLevel(obs_dim, 1, hidden)
        self.actor_l1 = ActorLevel(obs_dim + 1, 1, hidden)
        self.actor_l2 = ActorLevel(obs_dim + 2, rest_dim, hidden)

        # Optional independent binary head.
        if has_fire_head:
            self.actor_fire = build_fire_head(obs_dim)
        else:
            self.actor_fire = None

        # Gate covers the flight axes only.
        self.gate = GateNetwork(obs_dim, self.flight_dim, gate_hidden)

        # Targets.
        self.actor_l0_target = copy.deepcopy(self.actor_l0)
        self.actor_l1_target = copy.deepcopy(self.actor_l1)
        self.actor_l2_target = copy.deepcopy(self.actor_l2)
        self.gate_target = copy.deepcopy(self.gate)
        self.actor_fire_target = copy.deepcopy(self.actor_fire) if has_fire_head else None

        # Twin critic over the full action.
        self.critic = TwinCritic(obs_dim, act_dim, hidden)
        self.critic_target = copy.deepcopy(self.critic)

        # Independent optimizers per head — preserves per-head Adam state.
        self.opt_l0 = torch.optim.Adam(self.actor_l0.parameters(), lr=lr_actor)
        self.opt_l1 = torch.optim.Adam(self.actor_l1.parameters(), lr=lr_actor)
        self.opt_l2 = torch.optim.Adam(self.actor_l2.parameters(), lr=lr_actor)
        self.opt_gate = torch.optim.Adam(self.gate.parameters(), lr=lr_gate)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        if has_fire_head:
            self.opt_fire = torch.optim.Adam(self.actor_fire.parameters(), lr=lr_actor)
        else:
            self.opt_fire = None

        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        self.total_steps = 0
        self._update_count = 0

    def _compute_action(
        self,
        obs: torch.Tensor,
        actors: tuple | None = None,
        gate_net: nn.Module | None = None,
        fire_net: nn.Module | None = None,
    ) -> torch.Tensor:
        """Forward: flight = clip(anchor + delta * gate, -1, +1), fire = sigmoid head."""
        if actors is None:
            a0, a1, a2 = self.actor_l0, self.actor_l1, self.actor_l2
        else:
            a0, a1, a2 = actors
        if gate_net is None:
            gate_net = self.gate

        d_pitch = a0(obs)
        d_roll = a1(torch.cat([obs, d_pitch], dim=-1))
        d_rest = a2(torch.cat([obs, d_pitch, d_roll], dim=-1))
        delta = torch.cat([d_pitch, d_roll, d_rest], dim=-1)

        g = gate_net(obs)
        anchor_flight = self.anchor[: self.flight_dim].unsqueeze(0).expand_as(delta)
        flight = (anchor_flight + delta * g).clamp(-1.0, 1.0)

        if self.has_fire_head:
            if fire_net is None:
                fire_net = self.actor_fire
            fire = fire_net(obs)
            return torch.cat([flight, fire], dim=-1)
        return flight

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action = self._compute_action(obs_t).squeeze(0).numpy()

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.flight_dim)
            action[: self.flight_dim] = np.clip(
                action[: self.flight_dim] + noise, -1.0, 1.0
            )
            if self.has_fire_head and np.random.random() < self.fire_explore_prob:
                action[-1] = 1.0 if np.random.random() > 0.5 else 0.0
            action = action.astype(np.float32)

        return action

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        """Store a transition. Mirror augmentation is applied if mirror_cfg was set."""
        self.buffer.add(obs, action, reward, next_obs, done, mirror_cfg=self.mirror_cfg)

    def train_step(self) -> dict | None:
        if self.buffer.size < self.learning_starts:
            return None

        self._update_count += 1
        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # Critic update.
        with torch.no_grad():
            next_action = self._compute_action(
                next_obs,
                actors=(
                    self.actor_l0_target,
                    self.actor_l1_target,
                    self.actor_l2_target,
                ),
                gate_net=self.gate_target,
                fire_net=self.actor_fire_target,
            )
            noise = (torch.randn_like(next_action) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            tq1, tq2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(tq1, tq2)
            target_value = reward + (1.0 - done) * self.gamma * target_q

        cq1, cq2 = self.critic(obs, action)
        critic_loss = F.mse_loss(cq1, target_value) + F.mse_loss(cq2, target_value)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        info = {"critic_loss": critic_loss.item()}

        # Delayed actor + gate update.
        if self._update_count % self.policy_delay == 0:
            full_action = self._compute_action(obs)
            actor_loss = -self.critic.q1_forward(obs, full_action).mean()

            self.opt_l0.zero_grad()
            self.opt_l1.zero_grad()
            self.opt_l2.zero_grad()
            self.opt_gate.zero_grad()
            if self.opt_fire is not None:
                self.opt_fire.zero_grad()
            actor_loss.backward()
            self.opt_l0.step()
            self.opt_l1.step()
            self.opt_l2.step()
            self.opt_gate.step()
            if self.opt_fire is not None:
                self.opt_fire.step()

            self._soft_update(self.actor_l0, self.actor_l0_target)
            self._soft_update(self.actor_l1, self.actor_l1_target)
            self._soft_update(self.actor_l2, self.actor_l2_target)
            self._soft_update(self.gate, self.gate_target)
            self._soft_update(self.critic, self.critic_target)
            if self.has_fire_head:
                self._soft_update(self.actor_fire, self.actor_fire_target)

            info["actor_loss"] = actor_loss.item()

        return info

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt: dict = {
            "actor_l0": self.actor_l0.state_dict(),
            "actor_l1": self.actor_l1.state_dict(),
            "actor_l2": self.actor_l2.state_dict(),
            "gate": self.gate.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_l0_target": self.actor_l0_target.state_dict(),
            "actor_l1_target": self.actor_l1_target.state_dict(),
            "actor_l2_target": self.actor_l2_target.state_dict(),
            "gate_target": self.gate_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "opt_l0": self.opt_l0.state_dict(),
            "opt_l1": self.opt_l1.state_dict(),
            "opt_l2": self.opt_l2.state_dict(),
            "opt_gate": self.opt_gate.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "total_steps": self.total_steps,
            "_update_count": self._update_count,
        }
        if self.has_fire_head:
            ckpt["actor_fire"] = self.actor_fire.state_dict()
            ckpt["actor_fire_target"] = self.actor_fire_target.state_dict()
            ckpt["opt_fire"] = self.opt_fire.state_dict()
        torch.save(ckpt, path)
        buf_path = path.replace(".pt", "_buffer.npz")
        self.buffer.save(buf_path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.actor_l0.load_state_dict(ckpt["actor_l0"])
        self.actor_l1.load_state_dict(ckpt["actor_l1"])
        self.actor_l2.load_state_dict(ckpt["actor_l2"])
        self.gate.load_state_dict(ckpt["gate"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_l0_target.load_state_dict(ckpt["actor_l0_target"])
        self.actor_l1_target.load_state_dict(ckpt["actor_l1_target"])
        self.actor_l2_target.load_state_dict(ckpt["actor_l2_target"])
        self.gate_target.load_state_dict(ckpt["gate_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if self.has_fire_head and "actor_fire" in ckpt:
            self.actor_fire.load_state_dict(ckpt["actor_fire"])
            self.actor_fire_target.load_state_dict(ckpt["actor_fire_target"])
        try:
            self.opt_l0.load_state_dict(ckpt["opt_l0"])
            self.opt_l1.load_state_dict(ckpt["opt_l1"])
            self.opt_l2.load_state_dict(ckpt["opt_l2"])
            self.opt_gate.load_state_dict(ckpt["opt_gate"])
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
            if self.has_fire_head and "opt_fire" in ckpt:
                self.opt_fire.load_state_dict(ckpt["opt_fire"])
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"[NOML] optimizer state mismatch, resetting ({e})")
            self.opt_l0 = torch.optim.Adam(self.actor_l0.parameters(), lr=3e-4)
            self.opt_l1 = torch.optim.Adam(self.actor_l1.parameters(), lr=3e-4)
            self.opt_l2 = torch.optim.Adam(self.actor_l2.parameters(), lr=3e-4)
            self.opt_gate = torch.optim.Adam(self.gate.parameters(), lr=3e-4)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            if self.has_fire_head:
                self.opt_fire = torch.optim.Adam(self.actor_fire.parameters(), lr=3e-4)
        self.total_steps = ckpt.get("total_steps", 0)
        self._update_count = ckpt.get("_update_count", 0)
        buf_path = path.replace(".pt", "_buffer.npz")
        if os.path.isfile(buf_path):
            try:
                self.buffer.load(buf_path)
                print(f"[NOML] replay buffer loaded: {self.buffer.size:,} transitions")
            except Exception as e:
                print(f"[NOML] replay buffer mismatch, skipped ({e})")
