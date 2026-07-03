"""DesignMap: the differentiable Phi(f) mapping for the Unitree G1.

Phi maps a normalized design vector f in [-1, 1]^D to
  1. MJX model field updates (masses, inertias, joint limits, gains, geometry),
  2. environment-level PD/actuation parameters (kp, kd, torque limits, action scale),
  3. per-joint / per-foot / global URMA description vectors.

Everything is pure JAX so that gradients flow from the critic's value
prediction back to f during Value-Gradient Design Search.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp
import mujoco
import numpy as np

from darwin.utils.urdf_utils import G1Meta

MIN_JOINT_RANGE_GAP = 0.05  # rad, keeps lower < upper after scaling


@flax.struct.dataclass
class EnvParams:
    """Per-actuator control parameters produced by Phi."""

    kp: jnp.ndarray            # (A,)
    kd: jnp.ndarray            # (A,)
    torque_limit: jnp.ndarray  # (A,)
    action_scale: jnp.ndarray  # (A,)


@flax.struct.dataclass
class Descriptions:
    """URMA embodiment description vectors (all entries roughly in [-1, 1])."""

    joints: jnp.ndarray  # (J, D_j)
    feet: jnp.ndarray    # (2, D_f)
    global_: jnp.ndarray  # (D_g,) design params not attached to any joint/foot


def _apply(mode: str, nominal, scale: float, f, minimum=None):
    if mode == "rel":
        value = nominal * (1.0 + scale * f)
    elif mode == "abs":
        value = nominal + scale * f
    else:
        raise ValueError(f"Unknown bound mode {mode!r}")
    if minimum is not None:
        value = jnp.maximum(value, minimum)
    return value


class DesignMap:
    """Callable Phi. Instantiate once from the compiled base model."""

    def __init__(self, model: mujoco.MjModel, meta: G1Meta, bounds: dict):
        self.meta = meta
        self.bounds = bounds
        J = len(meta.actuated_joint_ids)
        B = len(meta.body_ids)
        A = model.nu
        G = len(meta.foot_geom_ids)
        self.J, self.B, self.A, self.G = J, B, A, G

        jid = meta.actuated_joint_ids
        self.nominal = {
            "jnt_pos": model.jnt_pos.copy(),
            "jnt_range": model.jnt_range.copy(),
            "jnt_stiffness": model.jnt_stiffness.copy(),
            "jnt_axis": model.jnt_axis.copy(),
            "dof_armature": model.dof_armature.copy(),
            "dof_damping": model.dof_damping.copy(),
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "body_ipos": model.body_ipos.copy(),
            "body_pos": model.body_pos.copy(),
            "geom_size": model.geom_size.copy(),
            "geom_pos": model.geom_pos.copy(),
        }

        # Actuator -> joint mapping (1:1 for the G1 torque motors).
        self.act_joint = model.actuator_trnid[:, 0].astype(np.int32)
        act_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(A)
        ]
        kp = np.zeros(A)
        kd = np.zeros(A)
        for i, name in enumerate(act_names):
            for entry in bounds["nominal_gains"]:
                if entry["pattern"] in name:
                    kp[i], kd[i] = entry["kp"], entry["kd"]
                    break
            else:
                kp[i], kd[i] = 40.0, 1.0
        self.nominal_kp = kp
        self.nominal_kd = kd
        self.nominal_torque_limit = model.actuator_ctrlrange[:, 1].copy()
        self.nominal_action_scale = np.full(A, bounds["nominal_action_scale"])

        # Joint id -> row index in the actuated-joint arrays.
        joint_row = {int(j): r for r, j in enumerate(jid)}
        # Actuator row per actuated-joint row (aligns env params with d_j).
        self.act_of_joint = np.array(
            [int(np.where(self.act_joint == j)[0][0]) for j in jid], dtype=np.int32
        )
        # Child body of each actuated joint.
        self.joint_body = model.jnt_bodyid[jid].astype(np.int32)
        # Row of that body inside meta.body_ids (body_ids = 1..nbody-1 -> id-1).
        self.joint_body_row = self.joint_body - 1

        # Leg-length bodies: knee (thigh length) and ankle_pitch (shank length).
        def find_bodies(key):
            out = []
            for b in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if key in name:
                    out.append(b)
            return np.array(sorted(out), dtype=np.int32)

        self.knee_bodies = find_bodies("knee_link")          # (2,) left, right
        self.shank_child_bodies = find_bodies("ankle_pitch_link")
        if len(self.knee_bodies) != 2 or len(self.shank_child_bodies) != 2:
            raise ValueError("Could not locate knee/ankle_pitch bodies for leg scaling")

        # Foot geoms grouped by side, aligned with meta.foot_body_ids order.
        fg = meta.foot_geom_ids
        fg_body = model.geom_bodyid[fg]
        self.foot_geoms_by_side = [
            fg[fg_body == meta.foot_body_ids[0]],
            fg[fg_body == meta.foot_body_ids[1]],
        ]
        self.spheres_per_foot = len(self.foot_geoms_by_side[0])

        # ---- f layout ------------------------------------------------------
        self.segments = {}
        self.labels = []
        cursor = 0

        def add(name, count, label_fn):
            nonlocal cursor
            self.segments[name] = (cursor, count)
            self.labels.extend(label_fn(i) for i in range(count))
            cursor += count

        jnames = meta.actuated_joint_names
        bnames = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))
            for b in meta.body_ids
        ]
        add("jnt_pos", J * 3, lambda i: f"joint/{jnames[i // 3]}/pos_{'xyz'[i % 3]}")
        add("jnt_range", J * 2, lambda i: f"joint/{jnames[i // 2]}/range_{'lu'[i % 2]}")
        add("armature", J, lambda i: f"joint/{jnames[i]}/armature")
        add("damping", J, lambda i: f"joint/{jnames[i]}/damping")
        add("stiffness", J, lambda i: f"joint/{jnames[i]}/stiffness")
        add("mass", B, lambda i: f"link/{bnames[i]}/mass")
        add("inertia", B * 3, lambda i: f"link/{bnames[i // 3]}/inertia_{'xyz'[i % 3]}")
        add("ipos", B * 3, lambda i: f"link/{bnames[i // 3]}/com_{'xyz'[i % 3]}")
        add("kp", A, lambda i: f"actuator/{act_names[i]}/kp")
        add("kd", A, lambda i: f"actuator/{act_names[i]}/kd")
        add("torque_limit", A, lambda i: f"actuator/{act_names[i]}/torque_limit")
        add("action_scale", A, lambda i: f"actuator/{act_names[i]}/action_scale")
        add("foot_size", G, lambda i: f"geometry/foot_sphere_{i}/size")
        add("foot_offset_x", G, lambda i: f"geometry/foot_sphere_{i}/offset_x")
        add("thigh_length", 2, lambda i: f"geometry/{'left' if i == 0 else 'right'}_thigh/length")
        add("shank_length", 2, lambda i: f"geometry/{'left' if i == 0 else 'right'}_shank/length")
        self.dim = cursor

        # Fixed per-joint context for d_j, normalized to O(1).
        axis = self.nominal["jnt_axis"][jid]
        jpos = self.nominal["jnt_pos"][jid]
        jrange = self.nominal["jnt_range"][jid]
        child_mass = self.nominal["body_mass"][self.joint_body]
        ctx = np.concatenate(
            [
                axis,                                        # (J, 3)
                jpos / 0.3,                                  # (J, 3)
                jrange / np.pi,                              # (J, 2)
                (kp[self.act_of_joint] / 150.0)[:, None],    # (J, 1)
                (kd[self.act_of_joint] / 4.0)[:, None],      # (J, 1)
                (child_mass / child_mass.max())[:, None],    # (J, 1)
            ],
            axis=1,
        )
        self.joint_context = jnp.asarray(ctx)

        # Which leg (0/1) each foot-side chain belongs to, for d_f assembly.
        self.d_joint_dim = self.joint_context.shape[1] + 19
        self.d_foot_dim = 1 + 2 * self.spheres_per_foot + 2
        self.d_global_dim = 7

    # -------------------------------------------------------------------- #

    def seg(self, f, name, shape=None):
        start, count = self.segments[name]
        out = f[start:start + count]
        return out.reshape(shape) if shape is not None else out

    def __call__(self, f: jnp.ndarray):
        """f (D,) -> (model_updates dict, EnvParams, Descriptions)."""
        b = self.bounds
        meta = self.meta
        nom = {k: jnp.asarray(v) for k, v in self.nominal.items()}
        jid = meta.actuated_joint_ids
        dof = meta.dof_adr
        bid = meta.body_ids

        jb, lb, ab, gb = b["joint"], b["link"], b["actuator"], b["geometry"]

        # --- joints ---------------------------------------------------------
        jnt_pos = nom["jnt_pos"].at[jid].add(
            jb["pos_offset"]["scale"] * self.seg(f, "jnt_pos", (self.J, 3))
        )
        f_range = self.seg(f, "jnt_range", (self.J, 2))
        scaled = nom["jnt_range"][jid] * (1.0 + jb["range"]["scale"] * f_range)
        lo = jnp.minimum(scaled[:, 0], scaled[:, 1] - MIN_JOINT_RANGE_GAP)
        jnt_range = nom["jnt_range"].at[jid].set(jnp.stack([lo, scaled[:, 1]], axis=1))
        dof_armature = nom["dof_armature"].at[dof].set(
            _apply("rel", nom["dof_armature"][dof], jb["armature"]["scale"],
                   self.seg(f, "armature"), jb["armature"]["min"])
        )
        dof_damping = nom["dof_damping"].at[dof].set(
            _apply("rel", nom["dof_damping"][dof], jb["damping"]["scale"],
                   self.seg(f, "damping"), jb["damping"]["min"])
        )
        jnt_stiffness = nom["jnt_stiffness"].at[jid].set(
            _apply("abs", nom["jnt_stiffness"][jid], jb["stiffness"]["scale"],
                   self.seg(f, "stiffness"), jb["stiffness"]["min"])
        )

        # --- links ------------------------------------------------------------
        body_mass = nom["body_mass"].at[bid].set(
            _apply("rel", nom["body_mass"][bid], lb["mass"]["scale"],
                   self.seg(f, "mass"), lb["mass"]["min"])
        )
        body_inertia = nom["body_inertia"].at[bid].set(
            _apply("rel", nom["body_inertia"][bid], lb["inertia"]["scale"],
                   self.seg(f, "inertia", (self.B, 3)), lb["inertia"]["min"])
        )
        body_ipos = nom["body_ipos"].at[bid].add(
            lb["com_offset"]["scale"] * self.seg(f, "ipos", (self.B, 3))
        )

        # --- geometry ---------------------------------------------------------
        fg = meta.foot_geom_ids
        geom_size = nom["geom_size"].at[fg, 0].set(
            _apply("rel", nom["geom_size"][fg, 0], gb["foot_size"]["scale"],
                   self.seg(f, "foot_size"), gb["foot_size"]["min"])
        )
        geom_pos = nom["geom_pos"].at[fg, 0].add(
            gb["foot_offset_x"]["scale"] * self.seg(f, "foot_offset_x")
        )
        body_pos = nom["body_pos"]
        thigh = self.seg(f, "thigh_length")
        shank = self.seg(f, "shank_length")
        body_pos = body_pos.at[self.knee_bodies].set(
            nom["body_pos"][self.knee_bodies] * (1.0 + gb["thigh_length"]["scale"] * thigh)[:, None]
        )
        body_pos = body_pos.at[self.shank_child_bodies].set(
            nom["body_pos"][self.shank_child_bodies]
            * (1.0 + gb["shank_length"]["scale"] * shank)[:, None]
        )

        model_updates = {
            "jnt_pos": jnt_pos,
            "jnt_range": jnt_range,
            "jnt_stiffness": jnt_stiffness,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "body_pos": body_pos,
            "geom_size": geom_size,
            "geom_pos": geom_pos,
        }

        # --- actuation --------------------------------------------------------
        env_params = EnvParams(
            kp=_apply("rel", jnp.asarray(self.nominal_kp), ab["kp"]["scale"],
                      self.seg(f, "kp"), ab["kp"]["min"]),
            kd=_apply("rel", jnp.asarray(self.nominal_kd), ab["kd"]["scale"],
                      self.seg(f, "kd"), ab["kd"]["min"]),
            torque_limit=_apply("rel", jnp.asarray(self.nominal_torque_limit),
                                ab["torque_limit"]["scale"],
                                self.seg(f, "torque_limit"), ab["torque_limit"]["min"]),
            action_scale=_apply("rel", jnp.asarray(self.nominal_action_scale),
                                ab["action_scale"]["scale"],
                                self.seg(f, "action_scale"), ab["action_scale"]["min"]),
        )

        descriptions = self._descriptions(f)
        return model_updates, env_params, descriptions

    # -------------------------------------------------------------------- #

    def _descriptions(self, f: jnp.ndarray) -> Descriptions:
        """Assemble URMA description vectors directly from normalized f slices.

        Using the normalized coordinates keeps every entry in [-1, 1] and gives
        the critic an identity-strength gradient path back to f during VGDS.
        """
        J, row = self.J, self.joint_body_row
        act = self.act_of_joint

        var = jnp.concatenate(
            [
                self.seg(f, "jnt_pos", (J, 3)),
                self.seg(f, "jnt_range", (J, 2)),
                self.seg(f, "armature")[:, None],
                self.seg(f, "damping")[:, None],
                self.seg(f, "stiffness")[:, None],
                self.seg(f, "mass")[row][:, None],
                self.seg(f, "inertia", (self.B, 3))[row],
                self.seg(f, "ipos", (self.B, 3))[row],
                self.seg(f, "kp")[act][:, None],
                self.seg(f, "kd")[act][:, None],
                self.seg(f, "torque_limit")[act][:, None],
                self.seg(f, "action_scale")[act][:, None],
            ],
            axis=1,
        )
        d_joints = jnp.concatenate([self.joint_context, var], axis=1)

        n = self.spheres_per_foot
        sizes = self.seg(f, "foot_size", (2, n))
        offsets = self.seg(f, "foot_offset_x", (2, n))
        thigh = self.seg(f, "thigh_length")[:, None]
        shank = self.seg(f, "shank_length")[:, None]
        side = jnp.array([[1.0], [-1.0]])
        d_feet = jnp.concatenate([side, sizes, offsets, thigh, shank], axis=1)

        # Pelvis (root body, id 1) has no actuated joint: expose it globally.
        d_global = jnp.concatenate(
            [
                self.seg(f, "mass")[0:1],
                self.seg(f, "inertia", (self.B, 3))[0],
                self.seg(f, "ipos", (self.B, 3))[0],
            ]
        )
        return Descriptions(joints=d_joints, feet=d_feet, global_=d_global)
