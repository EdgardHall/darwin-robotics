"""G1 MJX environment tests: reset/step with per-env morphologies."""

import os
import sys

import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.envs.g1_environment import G1Env

CFG = yaml.safe_load(open("configs/training.yaml"))
BOUNDS = yaml.safe_load(open("configs/g1_design_space.yaml"))


def make_env():
    return G1Env(CFG, BOUNDS)


def test_reset_step_batched():
    env = make_env()
    N = 2
    rng = jax.random.PRNGKey(0)
    states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(
        jax.random.split(rng, N), 0.3
    )
    assert states.obs.o_g.shape == (N, 16)
    assert states.obs.o_j.shape == (N, env.num_joints, 3)
    assert states.f.shape == (N, env.design_map.dim)
    # Different envs must have different morphologies.
    assert not jnp.allclose(states.f[0], states.f[1])
    assert not jnp.allclose(
        states.model_updates["body_mass"][0], states.model_updates["body_mass"][1]
    )

    step = jax.jit(jax.vmap(env.step))
    for _ in range(3):
        action = 0.1 * jax.random.normal(rng, (N, env.num_joints))
        states = step(states, action)
        for leaf in jax.tree.leaves(states.obs):
            assert jnp.isfinite(leaf).all()
        assert jnp.isfinite(states.reward).all()
    assert (states.step_count == 3).all()


def test_standing_is_stable():
    """With zero actions from the default pose the nominal robot should not
    instantly fall (PD holds the pose for at least a few control steps)."""
    env = make_env()
    rng = jax.random.PRNGKey(1)
    state = jax.jit(env.reset_with_design)(rng, jnp.zeros(env.design_map.dim))
    step = jax.jit(env.step)
    zero = jnp.zeros(env.num_joints)
    for _ in range(25):  # 0.5 s
        state = step(state, zero)
    assert float(state.done) == 0.0, "nominal G1 fell over while standing"
    assert float(state.data.qpos[2]) > 0.5


def test_fixed_design_reset_keeps_f():
    env = make_env()
    f = 0.2 * jnp.ones(env.design_map.dim)
    state = jax.jit(env.reset_with_design)(jax.random.PRNGKey(2), f)
    assert jnp.allclose(state.f, f)


if __name__ == "__main__":
    for fn in [test_reset_step_batched, test_standing_is_stable,
               test_fixed_design_reset_keeps_f]:
        fn()
        print(f"PASS {fn.__name__}")
