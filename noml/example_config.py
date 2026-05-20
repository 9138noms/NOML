"""Example configuration for a generic 6-DoF fixed-wing aircraft.

Action layout (act_dim = 6, has_fire_head=True):
    0: pitch    (-1, +1)
    1: roll     (-1, +1)
    2: yaw      (-1, +1)
    3: throttle (-1, +1)   maps to (0, 1) externally
    4: brake    (-1, +1)   maps to (0, 1) externally
    5: fire     (0, 1)     binary, sigmoid

Observation layout (illustrative; adapt to your env):
    [0:3]   velocity (body frame, normalized)
    [3:6]   forward vector (body frame)
    [6:9]   up vector (body frame)
    [9]     speed (normalized)
    [10:12] altitude / radar altitude
    [12]    pitch
    [13]    roll
    [14:16] yaw sin/cos
    [16]    throttle (last)
    [17]    fuel
    [18]    gear state
    [19]    alive flag
    [20]    on-ground flag
    [21:23] additional state slots
    [23:25] timing / context slots

Mirror indices below assume the layout above. If you change the layout, update
both `MIRROR_OBS_NEGATE` and `MIRROR_ACT_NEGATE` accordingly. Get this wrong and
the policy will train confidently incorrect behavior on half its data — worth a
unit test: `env.step(mirror(obs)) == mirror(env.step(obs))`.
"""

from __future__ import annotations

import numpy as np

from .mirror import MirrorConfig


OBS_DIM = 25
ACT_DIM = 6


# Anchor: wings level, near-MIL throttle, no brake, no fire.
# anchor[3] = 0.74 maps to throttle ~0.87 if you externally remap (x+1)/2.
ANCHOR = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0], dtype=np.float32)


# Indices whose sign flips under left-right reflection (example layout).
# Components along the body-x axis flip; components along body-y/z do not.
EXAMPLE_MIRROR_CFG = MirrorConfig(
    obs_negate=[
        0,   # velocity.x
        3,   # forward.x
        6,   # up.x
        13,  # roll
        14,  # yaw sin
        # Add your own angular-velocity / heading-error indices here.
    ],
    act_negate=[
        1,   # roll
        2,   # yaw
    ],
)


DEFAULT_HYPERPARAMS: dict = {
    "obs_dim": OBS_DIM,
    "act_dim": ACT_DIM,
    "anchor": ANCHOR,
    "mirror_cfg": EXAMPLE_MIRROR_CFG,
    "has_fire_head": True,
    "hidden": [256, 256],
    "gate_hidden": [128, 128],
    "lr_actor": 3e-4,
    "lr_gate": 3e-4,
    "lr_critic": 3e-4,
    "gamma": 0.99,
    "tau": 0.002,
    "policy_delay": 2,
    "target_noise": 0.2,
    "noise_clip": 0.5,
    "exploration_noise": 0.1,
    "fire_explore_prob": 0.3,
    "buffer_size": 500_000,
    "batch_size": 256,
    "learning_starts": 25_000,
}
