# NOML

> Hierarchical TD3 with Anchor Policy and Mirror Learning — a structural fix for continuous-control RL on tasks where one axis dominates the dynamics and pure TD3 collapses into oscillation.

**Test video**: https://www.youtube.com/watch?v=ZNn6wo_PX8Y

## TL;DR

```
action = clip(anchor + delta · gate, -1, +1)    (continuous axes)
fire   = σ(MLP(obs))                            (optional binary axis)
```

- **Anchor** — a fixed "safe" action (e.g., wings level, MIL throttle). The policy can never fully forget how to fly straight.
- **Delta** — a *hierarchical* TD3 actor split into three levels so pitch decisions inform roll, and pitch+roll inform the rest.
- **Gate** — a sigmoid network that learns how far to deviate from the anchor per axis.
- **Mirror Learning** — every transition is mirrored left↔right and added to the buffer, doubling sample count for free under symmetric dynamics.

Everything else (twin critic, target policy smoothing, policy delay, soft updates) is standard TD3.

## The problem with vanilla TD3 on flight control

Three pain points that pure TD3 has on continuous flight control:

1. **Catastrophic forgetting of the basics.** After a peak around 1.5M steps the policy starts oscillating on pitch, drifts, then crashes. Once it forgets level flight, it never recovers cleanly.
2. **Axis coupling.** Pitch is by far the dominant axis. Vanilla TD3's monolithic MLP treats all outputs as a flat vector — roll updates can perturb pitch even when pitch was already correct.
3. **Sample efficiency.** Real-environment RL is slow. Mirror symmetry is free data that monolithic networks don't exploit.

NOML attacks all three structurally rather than via reward shaping.

## Architecture

### 1. Anchor Policy

```python
ANCHOR = [0, 0, 0, 0.74, 0, 0]
#         pitch roll yaw thr brake fire
```

The full flight action is

```
flight = clip(ANCHOR + delta · gate, -1, +1)
```

If `gate → 0`, the policy reduces to the anchor. If `gate → 1`, the policy is free to override entirely. The critical property: even a partially-collapsed policy can't actively *fight* level flight — the worst it can do is let the anchor through.

### 2. Hierarchical Delta Actor

Three independent MLPs, each with its own optimizer:

```
L0: obs(D)           →  d_pitch  (1)
L1: obs(D) + d_pitch →  d_roll   (1)
L2: obs(D) + d_pitch + d_roll → d_yaw, d_throttle, d_brake  (3)
```

Each level conditions on the previous level's output, so roll "sees" the pitch decision before committing, and the rest of the surfaces see both. Independent optimizers stop a roll-side gradient update from corrupting the pitch head — empirically this is what kills the post-1.5M oscillation in flat TD3.

### 3. Gate Network

```
gate(obs) → [g_pitch, g_roll, g_yaw, g_throttle, g_brake]   ∈ [0, 1]^5
```

A small `[128, 128]` MLP with sigmoid output, learned jointly via the critic's policy gradient. The gate is the bridge between "always safe" and "fully expressive": the actor learns *how much* to deviate per axis per state.

In practice the gate tends to saturate near 1.0 once the policy is confident, which means the anchor only matters during recovery from instability — exactly when you want it.

### 4. Independent Fire Head (optional)

For tasks with a binary axis (e.g., fire / no-fire), it sits outside the anchor/gate machinery:

```
fire = σ(MLP(obs))
```

Exploration is ε-greedy random flip instead of Gaussian noise on the sigmoid output, because adding Gaussian noise to a binary decision is mostly wasted.

### 5. Mirror Learning

Under left-right symmetric dynamics, every transition stored in the replay buffer gets a mirrored twin. Reward and done are unchanged (reward must be symmetric). Every real transition becomes two — 2× effective sample size without any extra environment steps.

The mirror sign-flip indices are environment-specific and supplied by the user via `MirrorConfig`. See `noml/example_config.py` for a generic baseline.

Asymmetric wind, prevailing turn directions, or asymmetric loadouts would break the assumption — disable mirroring in those cases.

### 6. Critic

Standard TD3 twin critic over the full action. The actor's policy gradient `-Q1(obs, π(obs)).mean()` propagates back through *all heads* (L0, L1, L2, gate, optional fire) in a single backward pass — they share the loss but step their own Adam states.

## Hyperparameters that matter

| Knob | Default |
|---|---|
| γ | 0.99 |
| τ (soft update) | 0.002 |
| policy_delay | 2 |
| target_noise / clip | 0.2 / 0.5 |
| exploration_noise | 0.1 (Gaussian, flight axes) |
| fire exploration | ε=0.3 random flip |
| lr (all heads) | 3e-4 |
| batch | 256 |
| learning_starts | 25,000 |
| buffer | 500,000 |
| hidden | [256, 256] actors, [128, 128] gate |

τ = 0.002 is slower than the TD3 default 0.005 — necessary because the hierarchical structure means small target shifts compound across L0 → L1 → L2.

## Usage

```python
import numpy as np
from noml import NOML, MirrorConfig

agent = NOML(
    obs_dim=25,
    act_dim=6,
    anchor=np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0], dtype=np.float32),
    mirror_cfg=MirrorConfig(obs_negate=[0, 3, 6, 13, 14], act_negate=[1, 2]),
    has_fire_head=True,
)

for episode in range(N):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs, add_noise=True)
        next_obs, reward, done, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, float(done))
        agent.train_step()
        agent.total_steps += 1
        obs = next_obs

agent.save("noml_checkpoint.pt")
```

See `noml/example_config.py` for a more complete example configuration.

## What to watch out for

- **Gate saturation.** If the gate immediately pegs at 1.0 and stays there, the anchor becomes inert — you have a regular hierarchical TD3 with the anchor as a safety net during recovery. Log the per-axis gate mean.
- **Hierarchy order is task-specific.** Pitch → roll → rest assumes pitch is the dominant axis. If your environment has a different dominant axis, reorder L0/L1/L2.
- **Mirror correctness.** Get the negate index list wrong and the policy will learn confidently incorrect behavior on half its data. Worth unit-testing: `step(mirror(obs)) == mirror(step(obs))` should hold.
- **Independent optimizers, shared critic gradient.** All heads share the same actor_loss but step their own Adam states. This is intentional — it preserves per-head momentum estimates so a noisy update on one head doesn't desynchronize the others.

## Inspiration

- Heron Systems' AlphaDogfight (hierarchical control)
- LAG / JSBSim hierarchical RL papers
- The general observation that most air-combat RL uses PPO and TD3 is underexplored — partly because pure TD3 *does* collapse on these tasks, but with the right structure it doesn't have to.

## Citation

If you use NOML in your work, please cite it (see `CITATION.cff` for machine-readable form):

```
9138noms. NOML: Hierarchical TD3 with Anchor Policy and Mirror Learning. 2026.
https://github.com/9138noms/NOML
```

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
