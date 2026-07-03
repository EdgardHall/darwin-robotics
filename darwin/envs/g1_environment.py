"""MJX velocity-tracking locomotion environment for randomized G1 morphologies.

Functional, single-environment API (reset/step) designed to be `jax.vmap`-ed
by the trainer. Every reset draws a fresh design vector f, so each of the
parallel environments simulates a different G1 variant.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from darwin.envs import domain_randomization as dr
from darwin.envs import reward_functions as rf
from darwin.envs.control_functions import action_to_target, pd_torque
from darwin.models.design_map import DesignMap, Descriptions, EnvParams
from darwin.utils.urdf_utils import default_pose, load_g1_model

GAIT_PERIOD = 0.8  # s, phase clock for o_g
CONTACT_THRESHOLD = 0.01  # m, foot sphere bottom below this = ground contact


@flax.struct.dataclass
class Observation:
    o_g: jnp.ndarray  # (D_g,)   general observations
    o_j: jnp.ndarray  # (J, 3)   per-joint observations
    o_f: jnp.ndarray  # (2, 3)   per-foot observations


@flax.struct.dataclass
class EnvState:
    data: mjx.Data
    model_updates: dict
    env_params: EnvParams
    desc: Descriptions
    f: jnp.ndarray
    obs: Observation
    reward: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray
    command: jnp.ndarray
    last_action: jnp.ndarray
    last_qd: jnp.ndarray
    feet_air_time: jnp.ndarray
    step_count: jnp.ndarray
    ep_return: jnp.ndarray
    metrics: dict


def _quat_rotate_inv(q, v):
    """Rotate world vector v into the frame of quaternion q = (w, x, y, z)."""
    w, xyz = q[0], q[1:]
    t = 2.0 * jnp.cross(v, xyz)
    return v + w * t + jnp.cross(t, xyz)


class G1Env:
    def __init__(self, cfg: dict, design_bounds: dict):
        env_cfg = cfg["env"]
        self.cfg = env_cfg
        self.reward_cfg = cfg["reward"]
        self.ctrl_dt = float(env_cfg["ctrl_dt"])
        self.sim_dt = float(env_cfg["sim_dt"])
        self.decimation = int(round(self.ctrl_dt / self.sim_dt))
        self.episode_length = int(env_cfg["episode_length"])

        self.model, self.meta = load_g1_model(env_cfg["xml_path"], self.sim_dt)
        self.design_map = DesignMap(self.model, self.meta, design_bounds)
        self.mjx_model = mjx.put_model(self.model)
        self._data_template = mjx.make_data(self.mjx_model)

        self.default_qpos = jnp.asarray(
            default_pose(self.model, self.meta, env_cfg["base_init_height"])
        )
        self.default_joint_pos = self.default_qpos[self.meta.qpos_adr]
        self.num_joints = len(self.meta.actuated_joint_ids)
        self.act_of_joint = self.design_map.act_of_joint

        # Foot geoms grouped (2, n_spheres) for contact detection.
        self.foot_geoms = jnp.asarray(
            jnp.stack([jnp.asarray(g) for g in self.design_map.foot_geoms_by_side])
        )
        self.obs_dims = {
            "o_g": 16,
            "o_j": (self.num_joints, 3),
            "o_f": (2, 3),
        }

    # ------------------------------------------------------------------ #

    def _model_with(self, model_updates: dict):
        return self.mjx_model.tree_replace(model_updates)

    def _observe(self, data, env_state_bits):
        (command, last_action, feet_air_time, step_count, model_updates) = env_state_bits
        quat = data.qpos[3:7]
        lin_vel = _quat_rotate_inv(quat, data.qvel[0:3])
        ang_vel = data.qvel[3:6]  # already in base frame for a free joint
        gravity = _quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))

        t = step_count * self.ctrl_dt
        phase = 2.0 * jnp.pi * t / GAIT_PERIOD
        clock = jnp.array([jnp.sin(phase), jnp.cos(phase),
                           jnp.sin(phase + jnp.pi), jnp.cos(phase + jnp.pi)])
        o_g = jnp.concatenate([gravity, ang_vel, lin_vel, command, clock])

        q = data.qpos[self.meta.qpos_adr] - self.default_joint_pos
        qd = data.qvel[self.meta.dof_adr]
        o_j = jnp.stack([q, qd, last_action], axis=1)

        foot_z = self._foot_height(data, model_updates)
        contact = (foot_z < CONTACT_THRESHOLD).astype(jnp.float32)
        o_f = jnp.stack([contact, foot_z, feet_air_time], axis=1)
        return Observation(o_g=o_g, o_j=o_j, o_f=o_f), lin_vel, ang_vel, gravity, contact

    def _foot_height(self, data, model_updates):
        """Lowest point of each foot's contact spheres above the floor."""
        sizes = model_updates["geom_size"][self.foot_geoms, 0]  # (2, n)
        centers = data.geom_xpos[self.foot_geoms, 2]            # (2, n)
        return jnp.min(centers - sizes, axis=1)

    # ------------------------------------------------------------------ #

    def reset(self, rng, curriculum_c) -> EnvState:
        rng, k_f = jax.random.split(rng)
        f = dr.sample_design(k_f, self.design_map.dim, curriculum_c)
        return self.reset_with_design(rng, f)

    def reset_with_design(self, rng, f) -> EnvState:
        """Reset with a FIXED design vector (evaluation / design search)."""
        rng, k_cmd, k_q = jax.random.split(rng, 3)
        model_updates, env_params, desc = self.design_map(f)
        model = self._model_with(model_updates)

        qpos = self.default_qpos.at[self.meta.qpos_adr].add(
            dr.joint_reset_noise(k_q, self.num_joints)
        )
        data = self._data_template.replace(
            qpos=qpos, qvel=jnp.zeros(self.model.nv), ctrl=jnp.zeros(self.model.nu)
        )
        data = mjx.forward(model, data)

        command = dr.sample_command(k_cmd, self.cfg["command_ranges"])
        zeros_j = jnp.zeros(self.num_joints)
        feet_air_time = jnp.zeros(2)
        obs, *_ = self._observe(
            data, (command, zeros_j, feet_air_time, jnp.array(0), model_updates)
        )
        return EnvState(
            data=data,
            model_updates=model_updates,
            env_params=env_params,
            desc=desc,
            f=f,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            truncated=jnp.array(0.0),
            command=command,
            last_action=zeros_j,
            last_qd=zeros_j,
            feet_air_time=feet_air_time,
            step_count=jnp.array(0),
            ep_return=jnp.array(0.0),
            metrics={"tracking_error": jnp.array(0.0)},
        )

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        model = self._model_with(state.model_updates)
        target = action_to_target(action, self.default_joint_pos, state.env_params)
        qpos_adr, dof_adr = self.meta.qpos_adr, self.meta.dof_adr
        act = self.act_of_joint

        def substep(data, _):
            q = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            tau = pd_torque(target, q, qd, state.env_params)
            data = data.replace(ctrl=jnp.zeros(self.model.nu).at[act].set(tau))
            data = mjx.step(model, data)
            return data, tau

        data, taus = jax.lax.scan(substep, state.data, None, length=self.decimation)
        tau = taus[-1]

        step_count = state.step_count + 1
        bits = (state.command, action, state.feet_air_time, step_count, state.model_updates)
        obs, lin_vel, ang_vel, gravity, contact = self._observe(data, bits)

        # Feet air time bookkeeping (per control step).
        air_time = state.feet_air_time + self.ctrl_dt
        first_contact = (state.feet_air_time > 0.0) * contact
        feet_air_time = air_time * (1.0 - contact)

        qd = data.qvel[dof_adr]
        rcfg = self.reward_cfg
        terms = {
            "lin_vel_xy_tracking": rf.lin_vel_xy_tracking(lin_vel, state.command, rcfg["tracking_sigma"]),
            "ang_vel_z_tracking": rf.ang_vel_z_tracking(ang_vel, state.command, rcfg["tracking_sigma"]),
            "lin_vel_z": rf.lin_vel_z(lin_vel),
            "ang_vel_xy": rf.ang_vel_xy(ang_vel),
            "torques": rf.torques(tau),
            "action_rate": rf.action_rate(action, state.last_action),
            "joint_acc": rf.joint_acc(qd, state.last_qd, self.ctrl_dt),
            "feet_air_time": rf.feet_air_time(
                air_time, first_contact, state.command, rcfg["feet_air_time_target"]
            ),
            "orientation": rf.orientation(gravity),
            "base_height": rf.base_height(data.qpos[2], rcfg["target_base_height"]),
        }
        reward, _ = rf.compute_reward(rcfg["scales"], terms)
        reward = reward * self.ctrl_dt  # scale like Rudin et al.

        term_cfg = self.cfg["termination"]
        fallen = (
            (jnp.linalg.norm(gravity[:2]) > term_cfg["gravity_xy_max"])
            | (data.qpos[2] < term_cfg["base_height_min"])
        )
        truncated = (step_count >= self.episode_length) & ~fallen
        done = (fallen | truncated).astype(jnp.float32)

        tracking_error = jnp.linalg.norm(state.command[:2] - lin_vel[:2])
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            truncated=truncated.astype(jnp.float32),
            last_action=action,
            last_qd=qd,
            feet_air_time=feet_air_time,
            step_count=step_count,
            ep_return=state.ep_return + reward,
            metrics={"tracking_error": tracking_error},
        )
