"""Mirror Learning — exploit left-right symmetry of the environment.

Every real transition is augmented with a mirrored twin (sign-flipped on selected
obs/action indices). Effective sample count is doubled with no extra env steps.

Only valid under true left-right symmetric dynamics. Asymmetric wind, prevailing
turn direction, or asymmetric loadouts will break the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MirrorConfig:
    """Defines which obs/action indices flip sign under left-right reflection.

    These indices are environment-specific. Examples for a 6-DoF aircraft:
        obs:    velocity.x, forward.x, up.x, roll, sin(yaw), angVel.y, angVel.z
        action: roll, yaw

    See `example_config.py` for a generic baseline you can adapt.
    """

    obs_negate: list[int] = field(default_factory=list)
    act_negate: list[int] = field(default_factory=list)


def mirror_obs(obs: np.ndarray, cfg: MirrorConfig) -> np.ndarray:
    m = obs.copy()
    for i in cfg.obs_negate:
        if i < len(m):
            m[i] = -m[i]
    return m


def mirror_action(action: np.ndarray, cfg: MirrorConfig) -> np.ndarray:
    m = action.copy()
    for i in cfg.act_negate:
        if i < len(m):
            m[i] = -m[i]
    return m
