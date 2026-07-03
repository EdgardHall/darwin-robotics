"""Velocity-tracking reward terms (Rudin et al., 2022 style).

Each term is a pure function; `compute_reward` assembles the weighted sum
from the `reward.scales` section of the training config and also returns the
unweighted components for logging.
"""

from __future__ import annotations

import jax.numpy as jnp


def lin_vel_xy_tracking(base_lin_vel, command, sigma):
    err = jnp.sum(jnp.square(command[:2] - base_lin_vel[:2]))
    return jnp.exp(-err / sigma)


def ang_vel_z_tracking(base_ang_vel, command, sigma):
    err = jnp.square(command[2] - base_ang_vel[2])
    return jnp.exp(-err / sigma)


def lin_vel_z(base_lin_vel):
    return jnp.square(base_lin_vel[2])


def ang_vel_xy(base_ang_vel):
    return jnp.sum(jnp.square(base_ang_vel[:2]))


def torques(tau):
    return jnp.sum(jnp.square(tau))


def action_rate(action, last_action):
    return jnp.sum(jnp.square(action - last_action))


def joint_acc(qd, last_qd, dt):
    return jnp.sum(jnp.square((qd - last_qd) / dt))


def feet_air_time(air_time, first_contact, command, target):
    """Reward long swing phases on touchdown, only when moving."""
    reward = jnp.sum((air_time - target) * first_contact)
    return reward * (jnp.linalg.norm(command[:2]) > 0.1)


def orientation(projected_gravity):
    return jnp.sum(jnp.square(projected_gravity[:2]))


def base_height(height, target):
    return jnp.square(height - target)


def compute_reward(scales: dict, terms: dict):
    """Weighted sum of unweighted `terms` using `scales`; returns (total, terms)."""
    total = 0.0
    for name, scale in scales.items():
        total = total + scale * terms[name]
    return total, terms
