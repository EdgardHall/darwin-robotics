"""Gradient-free design-search baselines on the same critic surrogate.

These optimize the identical objective as VGDS (mean ensemble value minus the
trust-region penalty) but without gradients, providing a controlled
comparison of the value-gradient signal itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _make_batched_objective(objective_fn, state_bank, batch_size, rng):
    idx = jax.random.choice(
        rng, jax.tree.leaves(state_bank)[0].shape[0], (batch_size,), replace=False
    )
    obs_batch = jax.tree.map(lambda x: x[idx], state_bank)
    fn = jax.jit(jax.vmap(lambda f: objective_fn(f, obs_batch)[0]))
    return fn


def random_search(objective_fn, state_bank, dim, rng, samples=2000, batch_size=256,
                  f_clip=1.0, chunk=64):
    """Uniform random sampling in [-f_clip, f_clip]^dim."""
    rng, k_obj = jax.random.split(rng)
    batched = _make_batched_objective(objective_fn, state_bank, batch_size, k_obj)
    best_f, best_v = None, -np.inf
    for _ in range(samples // chunk):
        rng, k = jax.random.split(rng)
        cands = jax.random.uniform(k, (chunk, dim), minval=-f_clip, maxval=f_clip)
        vals = np.asarray(batched(cands))
        i = int(vals.argmax())
        if vals[i] > best_v:
            best_v, best_f = float(vals[i]), cands[i]
    return best_f, best_v


def simple_es(objective_fn, state_bank, f_init, rng, iterations=100, population=32,
              sigma_init=0.2, elite_frac=0.25, batch_size=256, f_clip=1.0):
    """(mu, lambda) evolution strategy with diagonal covariance (CMA-ES-lite)."""
    rng, k_obj = jax.random.split(rng)
    batched = _make_batched_objective(objective_fn, state_bank, batch_size, k_obj)
    mean = np.asarray(f_init, dtype=np.float64)
    sigma = np.full_like(mean, sigma_init)
    n_elite = max(2, int(population * elite_frac))
    history = []
    for _ in range(iterations):
        rng, k = jax.random.split(rng)
        eps = np.asarray(jax.random.normal(k, (population, mean.size)))
        cands = np.clip(mean + eps * sigma, -f_clip, f_clip)
        vals = np.asarray(batched(jnp.asarray(cands)))
        elite = cands[np.argsort(vals)[-n_elite:]]
        mean = elite.mean(axis=0)
        sigma = 0.9 * sigma + 0.1 * elite.std(axis=0)
        history.append(float(vals.max()))
    return jnp.asarray(mean), history
