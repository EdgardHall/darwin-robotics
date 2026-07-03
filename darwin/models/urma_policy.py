"""URMA actor network (Unified Robot Morphology Architecture).

Per-joint observations are encoded with a shared encoder, aggregated with an
attention mechanism whose keys come from the embodiment description vectors,
and decoded back into per-joint actions by a shared decoder conditioned on
the descriptions.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from darwin.models.design_map import Descriptions


class MLP(nn.Module):
    features: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.features):
            x = nn.Dense(size)(x)
            if i < len(self.features) - 1 or self.activate_final:
                x = nn.elu(x)
        return x


class AttentionAggregation(nn.Module):
    """z = sum_j softmax(g_phi(d_j) / tau)_j * e_j with learnable temperature."""

    key_hidden: Sequence[int]

    @nn.compact
    def __call__(self, latents, descriptions):
        # latents: (..., J, L), descriptions: (..., J, D_d)
        scores = MLP(tuple(self.key_hidden) + (1,))(descriptions)[..., 0]  # (..., J)
        log_tau = self.param("log_tau", nn.initializers.zeros, ())
        alpha = nn.softmax(scores / jnp.exp(log_tau), axis=-1)
        return jnp.sum(alpha[..., None] * latents, axis=-2)


class URMAActor(nn.Module):
    joint_encoder_hidden: Sequence[int] = (256, 128)
    attention_key_hidden: Sequence[int] = (128, 64)
    core_hidden: Sequence[int] = (512, 256, 128)
    latent_dim: int = 64
    init_log_std: float = -0.7

    @nn.compact
    def __call__(self, o_g, o_j, o_f, desc: Descriptions):
        """Returns (action_mean (..., J), log_std (J,))."""
        e_j = MLP(tuple(self.joint_encoder_hidden) + (self.latent_dim,))(o_j)
        z_joints = AttentionAggregation(self.attention_key_hidden)(e_j, desc.joints)

        e_f = MLP((64, self.latent_dim))(o_f)
        z_feet = AttentionAggregation((64, 32))(e_f, desc.feet)

        d_g = jnp.broadcast_to(desc.global_, o_g.shape[:-1] + desc.global_.shape[-1:])
        core_in = jnp.concatenate([o_g, z_joints, z_feet, d_g], axis=-1)
        core = MLP(self.core_hidden, activate_final=True)(core_in)

        num_joints = o_j.shape[-2]
        core_b = jnp.broadcast_to(
            core[..., None, :], core.shape[:-1] + (num_joints, core.shape[-1])
        )
        dec_in = jnp.concatenate([core_b, desc.joints, e_j], axis=-1)
        mean = MLP((128, 64, 1))(dec_in)[..., 0]

        log_std = self.param(
            "log_std",
            lambda key, shape: jnp.full(shape, self.init_log_std),
            (num_joints,),
        )
        return mean, log_std
