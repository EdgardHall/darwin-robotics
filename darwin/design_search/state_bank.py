"""State bank collection for VGDS.

Rolls out the frozen policy on many random morphologies and stores a diverse
set of RAW observations. During design search the frozen critic evaluates
V(s, Phi(f)) on these states, with the description vectors coming from the
candidate design f (not from the design the state was collected under).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from darwin.envs.g1_environment import G1Env, Observation
from darwin.models.urma_policy import URMAActor


def collect_state_bank(
    env: G1Env,
    actor: URMAActor,
    actor_params,
    normalizer,
    rng,
    num_states: int,
    num_designs: int,
    rollout_length: int,
    design_range: float = 1.0,
) -> Observation:
    """Returns an Observation pytree with leading dimension num_states."""

    rng, k_reset = jax.random.split(rng)
    reset_batch = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    states = reset_batch(jax.random.split(k_reset, num_designs), design_range)

    @jax.jit
    def rollout(states, rng):
        def step_fn(carry, _):
            states, rng = carry
            rng, k_reset = jax.random.split(rng)
            obs_n = jax.vmap(normalizer.normalize)(states.obs)
            mean, _ = actor.apply(
                actor_params, obs_n.o_g, obs_n.o_j, obs_n.o_f, states.desc
            )
            next_states = jax.vmap(env.step)(states, mean)
            reset_states = jax.vmap(env.reset, in_axes=(0, None))(
                jax.random.split(k_reset, num_designs), design_range
            )
            done_b = next_states.done.astype(bool)

            def select(new, old):
                mask = done_b.reshape(done_b.shape + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, old)

            states = jax.tree.map(select, reset_states, next_states)
            return (states, rng), states.obs

        (_, _), all_obs = jax.lax.scan(step_fn, (states, rng), None, length=rollout_length)
        return all_obs  # (T, num_designs, ...)

    all_obs = rollout(states, rng)
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), all_obs)
    total = rollout_length * num_designs
    rng, k_pick = jax.random.split(rng)
    idx = jax.random.choice(k_pick, total, (min(num_states, total),), replace=False)
    return jax.tree.map(lambda x: x[idx], flat)
