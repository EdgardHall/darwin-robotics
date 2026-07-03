"""DesignMap unit tests: roundtrip validity of Phi (verification plan #1)."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.models.design_map import DesignMap
from darwin.utils.urdf_utils import load_g1_model

XML = "assets/g1/scene_mjx.xml"
BOUNDS = yaml.safe_load(open("configs/g1_design_space.yaml"))


def make_map():
    model, meta = load_g1_model(XML)
    return model, meta, DesignMap(model, meta, BOUNDS)


def test_dimension_in_plan_range():
    _, _, dm = make_map()
    assert 500 <= dm.dim <= 700, dm.dim
    assert len(dm.labels) == dm.dim


def test_zero_design_is_nominal():
    model, meta, dm = make_map()
    updates, env_params, _ = dm(jnp.zeros(dm.dim))
    np.testing.assert_allclose(updates["body_mass"], model.body_mass, rtol=1e-6)
    np.testing.assert_allclose(updates["jnt_range"], model.jnt_range, rtol=1e-6)
    np.testing.assert_allclose(updates["body_pos"], model.body_pos, rtol=1e-6)
    np.testing.assert_allclose(env_params.kp, dm.nominal_kp, rtol=1e-6)


def test_random_designs_stay_physical():
    model, meta, dm = make_map()
    for seed in range(5):
        f = jax.random.uniform(
            jax.random.PRNGKey(seed), (dm.dim,), minval=-1.0, maxval=1.0
        )
        updates, env_params, desc = dm(f)
        bid = meta.body_ids  # world body legitimately has zero mass
        assert (updates["body_mass"][bid] > 0).all()
        assert (updates["body_inertia"][bid] > 0).all()
        jr = updates["jnt_range"][meta.actuated_joint_ids]
        assert (jr[:, 0] < jr[:, 1]).all(), "inverted joint limits"
        assert (updates["dof_armature"] >= 0).all()
        assert (updates["dof_damping"] >= 0).all()
        assert (env_params.kp > 0).all()
        assert (env_params.torque_limit > 0).all()
        fg = meta.foot_geom_ids
        assert (updates["geom_size"][fg, 0] > 0).all()
        for leaf in jax.tree.leaves(desc):
            assert jnp.isfinite(leaf).all()


def test_description_gradients_flow():
    _, _, dm = make_map()

    def desc_norm(f):
        _, _, desc = dm(f)
        return sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(desc))

    g = jax.grad(desc_norm)(0.1 * jnp.ones(dm.dim))
    assert jnp.isfinite(g).all()
    # Every parameter that appears in a description vector must have gradient.
    for name in ("kp", "mass", "foot_size", "thigh_length", "jnt_pos"):
        start, count = dm.segments[name]
        assert jnp.abs(g[start:start + count]).max() > 0, name


if __name__ == "__main__":
    for fn in [test_dimension_in_plan_range, test_zero_design_is_nominal,
               test_random_designs_stay_physical, test_description_gradients_flow]:
        fn()
        print(f"PASS {fn.__name__}")
