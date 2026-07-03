"""Direct-Design URMA critic with a K-head value ensemble.

Key modification versus the standard URMA critic (Sec. 3.2 of the plan):
the per-joint encoder receives BOTH the observations o_j AND the description
vectors d_j, so embodiment information flows through the attention *values*
(not only the keys). This yields much stronger, better-conditioned gradients
d V / d f for design search.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from darwin.models.design_map import Descriptions
from darwin.models.urma_policy import MLP, AttentionAggregation


class URMACritic(nn.Module):
    joint_encoder_hidden: Sequence[int] = (256, 128)
    attention_key_hidden: Sequence[int] = (128, 64)
    core_hidden: Sequence[int] = (512, 256, 128)
    latent_dim: int = 64
    num_heads: int = 5

    @nn.compact
    def __call__(self, o_g, o_j, o_f, desc: Descriptions):
        """Returns per-head values (..., K)."""
        # Direct-design encoder: observations concatenated with descriptions.
        d_j = jnp.broadcast_to(desc.joints, o_j.shape[:-1] + desc.joints.shape[-1:])
        e_j = MLP(tuple(self.joint_encoder_hidden) + (self.latent_dim,))(
            jnp.concatenate([o_j, d_j], axis=-1)
        )
        z_joints = AttentionAggregation(self.attention_key_hidden)(e_j, d_j)

        d_f = jnp.broadcast_to(desc.feet, o_f.shape[:-1] + desc.feet.shape[-1:])
        e_f = MLP((64, self.latent_dim))(jnp.concatenate([o_f, d_f], axis=-1))
        z_feet = AttentionAggregation((64, 32))(e_f, d_f)

        d_g = jnp.broadcast_to(desc.global_, o_g.shape[:-1] + desc.global_.shape[-1:])
        core_in = jnp.concatenate([o_g, z_joints, z_feet, d_g], axis=-1)
        core = MLP(self.core_hidden, activate_final=True)(core_in)

        values = [MLP((64, 1))(core)[..., 0] for _ in range(self.num_heads)]
        return jnp.stack(values, axis=-1)

    @staticmethod
    def mean_value(values):
        """V_bar = ensemble mean over the K heads."""
        return jnp.mean(values, axis=-1)
