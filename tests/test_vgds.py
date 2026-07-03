"""VGDS end-to-end test on a randomly initialized critic (no training)."""

import os
import sys

import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.design_search.vgds import make_value_fn, vgds
from darwin.envs.g1_environment import G1Env, Observation
from darwin.models.urma_critic import URMACritic
from darwin.training.ppo_trainer import Normalizer

CFG = yaml.safe_load(open("configs/training.yaml"))
BOUNDS = yaml.safe_load(open("configs/g1_design_space.yaml"))


def test_vgds_improves_surrogate():
    env = G1Env(CFG, BOUNDS)
    dm = env.design_map
    rng = jax.random.PRNGKey(0)

    # Small synthetic state bank (shapes as produced by the env).
    M, J = 64, env.num_joints
    ks = jax.random.split(rng, 3)
    bank = Observation(
        o_g=0.5 * jax.random.normal(ks[0], (M, 16)),
        o_j=0.5 * jax.random.normal(ks[1], (M, J, 3)),
        o_f=0.5 * jax.random.normal(ks[2], (M, 2, 3)),
    )

    critic = URMACritic()
    _, _, desc0 = dm(jnp.zeros(dm.dim))
    params = critic.init(
        jax.random.PRNGKey(1), bank.o_g[0], bank.o_j[0], bank.o_f[0], desc0
    )
    normalizer = Normalizer.create(env.obs_dims)
    f_ref = jnp.zeros(dm.dim)

    f_opt, history = vgds(
        critic, params, normalizer, dm, bank,
        f_init=f_ref, f_ref=f_ref, rng=jax.random.PRNGKey(2),
        iterations=30, batch_size=32, learning_rate=0.02,
        trust_region_lambda=1.0, delta_max=0.05, f_clip=1.0,
    )
    assert jnp.isfinite(f_opt).all()
    assert (jnp.abs(f_opt) <= 1.0).all()
    assert jnp.abs(f_opt - f_ref).max() > 0, "VGDS did not move the design"
    assert all(jnp.isfinite(jnp.asarray(history["value"])))
    assert max(history["grad_norm"]) > 0

    # The surrogate objective should have improved from the starting design.
    value_fn = jax.jit(make_value_fn(critic, params, normalizer, dm))
    v0 = float(value_fn(f_ref, bank))
    v1 = float(value_fn(f_opt, bank))
    assert v1 > v0, f"V̄ did not improve: {v0:.4f} -> {v1:.4f}"
    print(f"surrogate value improved {v0:.4f} -> {v1:.4f}")


if __name__ == "__main__":
    test_vgds_improves_surrogate()
    print("PASS test_vgds_improves_surrogate")
