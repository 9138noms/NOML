"""Network modules: hierarchical actors, gate, twin critic."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorLevel(nn.Module):
    """One level of the hierarchical actor. Outputs delta in [-1, +1] via Tanh."""

    def __init__(self, input_dim: int, output_dim: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateNetwork(nn.Module):
    """Per-axis gate in [0, 1]. Controls how far the policy deviates from the anchor."""

    def __init__(self, input_dim: int, output_dim: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [128, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinCritic(nn.Module):
    """Standard TD3 twin Q-network. min(Q1, Q2) is used for target."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        inp = obs_dim + act_dim

        def make_q() -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = inp
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


def build_fire_head(obs_dim: int, hidden: int = 128) -> nn.Sequential:
    """Optional independent binary head (e.g., a 'fire' action). Sigmoid output."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )
