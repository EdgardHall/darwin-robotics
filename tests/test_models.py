"""URMA actor/critic forward-pass tests."""

import os
import sys

import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.models.design_map import DesignMap
from darwin.models.urma_critic import URMACritic
from darwin.models.urma_policy import URMAActor
from darwin.utils.urdf_utils import load_g1_model

BOUNDS = yaml.safe_load(open("configs/g1_design_space.yaml"))


def setup():
    model, meta, = load_g1_model("assets/g1/scene_mjx.xml")
    dm = DesignMap(model, meta, BOUNDS)
    _, _, desc = dm(jnp.zeros(dm.dim))
    J = dm.J
    o_g = jnp.ones(16)
    o_j = jnp.ones((J, 3))
    o_f = jnp.ones((2, 3))
    return dm, desc, o_g, o_j, o_f, J


def test_actor_forward():
    dm, desc, o_g, o_j, o_f, J = setup()
    actor = URMAActor()
    params = actor.init(jax.random.PRNGKey(0), o_g, o_j, o_f, desc)
    mean, log_std = actor.apply(params, o_g, o_j, o_f, desc)
    assert mean.shape == (J,) and log_std.shape == (J,)
    # Batched call with batched descriptions.
    B = 4
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), desc)
    mean_b, _ = actor.apply(
        params,
        jnp.broadcast_to(o_g, (B, 16)),
        jnp.broadcast_to(o_j, (B, J, 3)),
        jnp.broadcast_to(o_f, (B, 2, 3)),
        batch,
    )
    assert mean_b.shape == (B, J)
    assert jnp.isfinite(mean_b).all()


def test_critic_forward_and_ensemble():
    dm, desc, o_g, o_j, o_f, J = setup()
    critic = URMACritic(num_heads=5)
    params = critic.init(jax.random.PRNGKey(0), o_g, o_j, o_f, desc)
    values = critic.apply(params, o_g, o_j, o_f, desc)
    assert values.shape == (5,)
    # Batched states with a SINGLE (unbatched) description: VGDS usage.
    M = 8
    values_b = critic.apply(
        params,
        jnp.broadcast_to(o_g, (M, 16)),
        jnp.broadcast_to(o_j, (M, J, 3)),
        jnp.broadcast_to(o_f, (M, 2, 3)),
        desc,
    )
    assert values_b.shape == (M, 5)
    assert jnp.isfinite(values_b).all()


def test_critic_gradient_wrt_design():
    """Verification plan #4: nabla_f V_bar is non-zero and finite."""
    model, meta = load_g1_model("assets/g1/scene_mjx.xml")
    dm = DesignMap(model, meta, BOUNDS)
    _, _, desc0 = dm(jnp.zeros(dm.dim))
    J = dm.J
    o_g, o_j, o_f = jnp.ones(16), jnp.ones((J, 3)), jnp.ones((2, 3))
    critic = URMACritic()
    params = critic.init(jax.random.PRNGKey(1), o_g, o_j, o_f, desc0)

    def v_bar(f):
        _, _, desc = dm(f)
        return jnp.mean(critic.apply(params, o_g, o_j, o_f, desc))

    for seed in range(3):
        f = 0.5 * jax.random.uniform(
            jax.random.PRNGKey(seed), (dm.dim,), minval=-1.0, maxval=1.0
        )
        g = jax.grad(v_bar)(f)
        assert jnp.isfinite(g).all()
        assert jnp.abs(g).max() > 0
        nonzero_frac = (jnp.abs(g) > 1e-12).mean()
        assert nonzero_frac > 0.5, f"only {nonzero_frac:.0%} of design dims get gradient"


if __name__ == "__main__":
    for fn in [test_actor_forward, test_critic_forward_and_ensemble,
               test_critic_gradient_wrt_design]:
        fn()
        print(f"PASS {fn.__name__}")
