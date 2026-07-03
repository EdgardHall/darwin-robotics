"""PD torque control (Rudin et al., 2022 style).

The G1 XML exposes pure torque motors; the PD controller lives here so that
its gains (kp, kd), torque limits and action scaling are part of the design
space optimized by VGDS.
"""

from __future__ import annotations

import jax.numpy as jnp

from darwin.models.design_map import EnvParams


def action_to_target(action, default_joint_pos, env_params: EnvParams):
    """Map a policy action in [-1, 1]^J to target joint positions."""
    return default_joint_pos + env_params.action_scale * jnp.clip(action, -1.0, 1.0)


def pd_torque(target_q, q, qd, env_params: EnvParams):
    """tau = kp (q_target - q) - kd qd, clipped to the design torque limits."""
    tau = env_params.kp * (target_q - q) - env_params.kd * qd
    return jnp.clip(tau, -env_params.torque_limit, env_params.torque_limit)
