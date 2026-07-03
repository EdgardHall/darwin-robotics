"""Value-Gradient Design Search (VGDS).

Maximizes
    J_lambda(f) = E_{s ~ B} [ V_bar(s, Phi(f)) ] - lambda * ||f - f_ref||^2 / D
by Adam ascent on the normalized design vector f, differentiating through the
frozen critic ensemble and the DesignMap. The simulator is never touched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from darwin.models.design_map import DesignMap


def make_value_fn(critic, critic_params, normalizer, design_map: DesignMap):
    """Returns V_bar(f, obs_batch): ensemble-mean value averaged over states."""

    def value_fn(f, obs_batch):
        _, _, desc = design_map(f)  # only the descriptions reach the critic
        obs_n = jax.vmap(normalizer.normalize)(obs_batch)
        values = critic.apply(
            critic_params, obs_n.o_g, obs_n.o_j, obs_n.o_f, desc
        )  # (M, K)
        return jnp.mean(values)

    return value_fn


def vgds(
    critic,
    critic_params,
    normalizer,
    design_map: DesignMap,
    state_bank,
    f_init: jnp.ndarray,
    f_ref: jnp.ndarray,
    rng,
    iterations: int = 500,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    trust_region_lambda: float = 100.0,
    delta_max: float = 0.05,
    f_clip: float = 1.0,
):
    """Run VGDS from f_init. Returns (f_opt, history dict)."""
    value_fn = make_value_fn(critic, critic_params, normalizer, design_map)
    dim = design_map.dim
    bank_size = jax.tree.leaves(state_bank)[0].shape[0]

    def objective(f, obs_batch):
        value = value_fn(f, obs_batch)
        penalty = trust_region_lambda * jnp.sum(jnp.square(f - f_ref)) / dim
        return value - penalty, value

    grad_fn = jax.grad(objective, has_aux=True)
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def step(f, opt_state, rng):
        rng, k_batch = jax.random.split(rng)
        idx = jax.random.choice(k_batch, bank_size, (batch_size,), replace=False)
        obs_batch = jax.tree.map(lambda x: x[idx], state_bank)
        g, value = grad_fn(f, obs_batch)
        # Ascent: feed -g to the (minimizing) optimizer.
        delta, opt_state = optimizer.update(-g, opt_state, f)
        delta = jnp.clip(delta, -delta_max, delta_max)
        f = jnp.clip(f + delta, -f_clip, f_clip)
        return f, opt_state, rng, value, jnp.linalg.norm(g)

    f = f_init
    opt_state = optimizer.init(f)
    history = {"value": [], "grad_norm": []}
    for _ in range(iterations):
        f, opt_state, rng, value, grad_norm = step(f, opt_state, rng)
        history["value"].append(float(value))
        history["grad_norm"].append(float(grad_norm))
    return f, history
