"""Model loading and MJX-oriented preprocessing for the Unitree G1.

The stock ``g1_29dof.xml`` from unitreerobotics/unitree_mujoco uses mesh
collision geometries everywhere, which is prohibitively expensive under MJX.
Following common practice (e.g. MuJoCo Playground), we keep only the foot
contact spheres active for collision and let everything else be visual.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

FOOT_BODY_KEYWORD = "ankle_roll"  # bodies whose geoms are the foot contact spheres


@dataclasses.dataclass(frozen=True)
class G1Meta:
    """Static index metadata extracted from the compiled model."""

    actuated_joint_ids: np.ndarray   # (29,) joint ids (hinges, excludes free joint)
    actuated_joint_names: list
    qpos_adr: np.ndarray             # (29,) qpos address per actuated joint
    dof_adr: np.ndarray              # (29,) dof address per actuated joint
    body_ids: np.ndarray             # (nbody-1,) all non-world bodies
    foot_body_ids: np.ndarray        # (2,) left/right ankle_roll bodies
    foot_geom_ids: np.ndarray        # foot contact sphere geom ids
    floor_geom_id: int
    torso_body_id: int               # pelvis / floating base body


def load_g1_model(xml_path: str, sim_dt: float = 0.002) -> tuple[mujoco.MjModel, G1Meta]:
    """Load the G1 scene, restrict collisions to feet + floor, set solver options."""
    model = mujoco.MjModel.from_xml_path(xml_path)

    model.opt.timestep = sim_dt
    model.opt.iterations = 8
    model.opt.ls_iterations = 8
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # MJX-friendly: disable Euler damping integration correction cost.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id < 0:
        raise ValueError(f"No 'floor' geom in {xml_path}; use the scene XML.")

    foot_body_ids = [
        b for b in range(model.nbody)
        if FOOT_BODY_KEYWORD in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
    ]
    if len(foot_body_ids) != 2:
        raise ValueError(f"Expected 2 foot bodies, found {len(foot_body_ids)}")

    foot_geom_ids = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        is_foot_sphere = (
            body in foot_body_ids
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
        if g == floor_geom_id:
            continue
        if is_foot_sphere:
            foot_geom_ids.append(g)
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1
        else:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0

    actuated = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    meta = G1Meta(
        actuated_joint_ids=np.array(actuated, dtype=np.int32),
        actuated_joint_names=[
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in actuated
        ],
        qpos_adr=np.array([model.jnt_qposadr[j] for j in actuated], dtype=np.int32),
        dof_adr=np.array([model.jnt_dofadr[j] for j in actuated], dtype=np.int32),
        body_ids=np.arange(1, model.nbody, dtype=np.int32),
        foot_body_ids=np.array(foot_body_ids, dtype=np.int32),
        foot_geom_ids=np.array(foot_geom_ids, dtype=np.int32),
        floor_geom_id=floor_geom_id,
        torso_body_id=1,
    )
    return model, meta


def default_pose(model: mujoco.MjModel, meta: G1Meta, base_height: float) -> np.ndarray:
    """A stable standing pose: slightly bent knees, arms at the side."""
    qpos = np.zeros(model.nq)
    qpos[2] = base_height
    qpos[3] = 1.0  # identity quaternion (w, x, y, z)
    joint_defaults = {
        "hip_pitch": -0.4,
        "knee": 0.8,
        "ankle_pitch": -0.4,
        "shoulder_roll_joint_left": 0.2,
    }
    for name, adr in zip(meta.actuated_joint_names, meta.qpos_adr):
        for key, value in joint_defaults.items():
            if key in name:
                qpos[adr] = value
                break
    # Slight outward shoulder roll so arms don't intersect the torso.
    for name, adr in zip(meta.actuated_joint_names, meta.qpos_adr):
        if "shoulder_roll" in name:
            qpos[adr] = 0.25 if name.startswith("left") else -0.25
    return qpos
