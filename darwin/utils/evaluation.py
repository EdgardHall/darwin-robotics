"""Rollout-based evaluation of a fixed design with the frozen policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def evaluate_design(env, actor, actor_params, normalizer, f, rng,
                    num_envs: int = 64, episode_length: int | None = None,
                    deterministic: bool = True):
    """Run full episodes with a FIXED design f. Returns metrics dict."""
    episode_length = episode_length or env.episode_length

    reset_batch = jax.vmap(env.reset_with_design, in_axes=(0, None))

    @jax.jit
    def run(rng):
        rng, k_reset = jax.random.split(rng)
        states = reset_batch(jax.random.split(k_reset, num_envs), f)

        def step_fn(carry, _):
            states, alive, ep_return, track_sq, steps = carry
            obs_n = jax.vmap(normalizer.normalize)(states.obs)
            mean, _ = actor.apply(
                actor_params, obs_n.o_g, obs_n.o_j, obs_n.o_f, states.desc
            )
            next_states = jax.vmap(env.step)(states, mean)
            ep_return = ep_return + alive * next_states.reward
            track_sq = track_sq + alive * jnp.square(
                next_states.metrics["tracking_error"]
            )
            steps = steps + alive
            alive = alive * (1.0 - next_states.done)
            return (next_states, alive, ep_return, track_sq, steps), None

        init = (
            states,
            jnp.ones(num_envs),
            jnp.zeros(num_envs),
            jnp.zeros(num_envs),
            jnp.zeros(num_envs),
        )
        (states, alive, ep_return, track_sq, steps), _ = jax.lax.scan(
            step_fn, init, None, length=episode_length
        )
        return ep_return, jnp.sqrt(track_sq / jnp.maximum(steps, 1.0)), steps

    ep_return, tracking_rmse, steps = run(rng)
    return {
        "mean_return": float(jnp.mean(ep_return)),
        "std_return": float(jnp.std(ep_return)),
        "tracking_rmse": float(jnp.mean(tracking_rmse)),
        "mean_episode_length": float(jnp.mean(steps)),
    }
