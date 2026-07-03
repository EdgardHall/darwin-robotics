"""Morphology randomization: design sampling for multi-embodiment training.

At every episode reset each parallel environment draws a fresh normalized
design vector f ~ U([-c, c]^D), where c is the curriculum half-width, and the
DesignMap turns it into a new G1 variant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_design(rng, dim: int, c) -> jnp.ndarray:
    """f ~ U([-c, c]^dim)."""
    return jax.random.uniform(rng, (dim,), minval=-1.0, maxval=1.0) * c


def sample_command(rng, ranges) -> jnp.ndarray:
    """Sample a (vx, vy, yaw_rate) command from the configured ranges."""
    lows = jnp.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_yaw"][0]])
    highs = jnp.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_yaw"][1]])
    return jax.random.uniform(rng, (3,), minval=lows, maxval=highs)


def joint_reset_noise(rng, num_joints: int, scale: float = 0.05) -> jnp.ndarray:
    return scale * jax.random.normal(rng, (num_joints,))
