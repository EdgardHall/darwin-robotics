"""Multi-embodiment PPO trainer for URMA on randomized G1 morphologies.

Self-contained JAX/Flax PPO (clipped objective, GAE, observation
normalization). Structure follows the RL-X / Rudin-style MJX pipelines: a
vmapped functional environment, brax-style auto-reset via `jnp.where`, and a
scan-based rollout, all jitted end to end.
"""

from __future__ import annotations

import functools
import os
import pickle
import time

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from darwin.envs.g1_environment import G1Env, Observation
from darwin.models.urma_critic import URMACritic
from darwin.models.urma_policy import URMAActor
from darwin.training.curriculum import Curriculum


@flax.struct.dataclass
class Normalizer:
    mean: Observation
    var: Observation
    count: jnp.ndarray

    @classmethod
    def create(cls, dims):
        zeros = Observation(
            o_g=jnp.zeros(dims["o_g"]),
            o_j=jnp.zeros(dims["o_j"][1]),
            o_f=jnp.zeros(dims["o_f"][1]),
        )
        ones = Observation(
            o_g=jnp.ones(dims["o_g"]),
            o_j=jnp.ones(dims["o_j"][1]),
            o_f=jnp.ones(dims["o_f"][1]),
        )
        return cls(mean=zeros, var=ones, count=jnp.array(1e-4))

    def normalize(self, obs: Observation) -> Observation:
        def norm(x, mean, var):
            return jnp.clip((x - mean) / jnp.sqrt(var + 1e-8), -10.0, 10.0)

        return Observation(
            o_g=norm(obs.o_g, self.mean.o_g, self.var.o_g),
            o_j=norm(obs.o_j, self.mean.o_j, self.var.o_j),
            o_f=norm(obs.o_f, self.mean.o_f, self.var.o_f),
        )

    def update(self, obs: Observation) -> "Normalizer":
        """Welford-style batched update; per-joint/per-foot stats are pooled."""

        def stats(x, feat_dim):
            flat = x.reshape(-1, feat_dim)
            return jnp.mean(flat, 0), jnp.var(flat, 0), flat.shape[0]

        new = {}
        for name, feat in (("o_g", None), ("o_j", None), ("o_f", None)):
            x = getattr(obs, name)
            feat_dim = x.shape[-1]
            b_mean, b_var, b_count = stats(x, feat_dim)
            mean = getattr(self.mean, name)
            var = getattr(self.var, name)
            delta = b_mean - mean
            tot = self.count + b_count
            new_mean = mean + delta * (b_count / tot)
            m_a = var * self.count
            m_b = b_var * b_count
            m2 = m_a + m_b + jnp.square(delta) * self.count * b_count / tot
            new[name] = (new_mean, m2 / tot)
        return Normalizer(
            mean=Observation(o_g=new["o_g"][0], o_j=new["o_j"][0], o_f=new["o_f"][0]),
            var=Observation(o_g=new["o_g"][1], o_j=new["o_j"][1], o_f=new["o_f"][1]),
            count=self.count + obs.o_g.reshape(-1, obs.o_g.shape[-1]).shape[0],
        )


@flax.struct.dataclass
class Transition:
    obs: Observation           # normalized
    desc: object               # Descriptions
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray         # ensemble mean V_bar
    reward: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray


def gaussian_log_prob(mean, log_std, action):
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * jnp.square((action - mean) / std) - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


class PPOTrainer:
    def __init__(self, env: G1Env, cfg: dict):
        self.env = env
        self.cfg = cfg
        ppo = cfg["ppo"]
        net = cfg["network"]
        self.num_envs = int(cfg["env"]["num_envs"])
        self.rollout_length = int(ppo["rollout_length"])
        self.actor = URMAActor(
            joint_encoder_hidden=net["joint_encoder_hidden"],
            attention_key_hidden=net["attention_key_hidden"],
            core_hidden=net["core_hidden"],
            latent_dim=net["latent_dim"],
            init_log_std=net["init_log_std"],
        )
        self.critic = URMACritic(
            joint_encoder_hidden=net["joint_encoder_hidden"],
            attention_key_hidden=net["attention_key_hidden"],
            core_hidden=net["core_hidden"],
            latent_dim=net["latent_dim"],
            num_heads=net["num_critic_heads"],
        )
        self.curriculum = Curriculum.from_config(cfg["curriculum"])

        steps_per_iter = self.num_envs * self.rollout_length
        self.num_iterations = max(1, int(float(ppo["total_env_steps"]) // steps_per_iter))
        self.steps_per_iter = steps_per_iter

        updates_per_iter = int(ppo["update_epochs"]) * int(ppo["num_minibatches"])
        if ppo.get("anneal_lr", True):
            schedule = optax.linear_schedule(
                float(ppo["learning_rate"]), 0.0, self.num_iterations * updates_per_iter
            )
        else:
            schedule = float(ppo["learning_rate"])
        self.tx = optax.chain(
            optax.clip_by_global_norm(float(ppo["max_grad_norm"])),
            optax.adam(schedule),
        )

        self._reset_batch = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        self._train_iter = jax.jit(self._train_iteration, donate_argnums=(0,))

    # ------------------------------------------------------------------ #

    def init_state(self, rng):
        rng, k_env, k_actor, k_critic = jax.random.split(rng, 4)
        env_states = self._reset_batch(
            jax.random.split(k_env, self.num_envs), self.curriculum.c
        )
        obs0 = jax.tree.map(lambda x: x[0], env_states.obs)
        desc0 = jax.tree.map(lambda x: x[0], env_states.desc)
        actor_params = self.actor.init(k_actor, obs0.o_g, obs0.o_j, obs0.o_f, desc0)
        critic_params = self.critic.init(k_critic, obs0.o_g, obs0.o_j, obs0.o_f, desc0)
        params = {"actor": actor_params, "critic": critic_params}
        opt_state = self.tx.init(params)
        normalizer = Normalizer.create(self.env.obs_dims)
        return {
            "rng": rng,
            "env_states": env_states,
            "params": params,
            "opt_state": opt_state,
            "normalizer": normalizer,
        }

    # ------------------------------------------------------------------ #

    def _policy_step(self, params, normalizer, env_states, rng):
        obs_n = jax.vmap(normalizer.normalize)(env_states.obs)
        mean, log_std = self.actor.apply(
            params["actor"], obs_n.o_g, obs_n.o_j, obs_n.o_f, env_states.desc
        )
        noise = jax.random.normal(rng, mean.shape)
        action = mean + jnp.exp(log_std) * noise
        log_prob = gaussian_log_prob(mean, log_std, action)
        values = self.critic.apply(
            params["critic"], obs_n.o_g, obs_n.o_j, obs_n.o_f, env_states.desc
        )
        return obs_n, action, log_prob, jnp.mean(values, axis=-1)

    def _train_iteration(self, carry, curriculum_c):
        params = carry["params"]
        opt_state = carry["opt_state"]
        normalizer = carry["normalizer"]
        ppo = self.cfg["ppo"]

        def rollout_step(scan_carry, _):
            env_states, rng = scan_carry
            rng, k_act, k_reset = jax.random.split(rng, 3)
            obs_n, action, log_prob, value = self._policy_step(
                params, normalizer, env_states, k_act
            )
            desc = env_states.desc
            next_states = jax.vmap(self.env.step)(env_states, action)

            finished_return = next_states.ep_return
            finished_mask = next_states.done
            tracking_error = next_states.metrics["tracking_error"]

            reset_states = jax.vmap(self.env.reset, in_axes=(0, None))(
                jax.random.split(k_reset, self.num_envs), curriculum_c
            )
            done_b = next_states.done.astype(bool)

            def select(new, old):
                mask = done_b.reshape(done_b.shape + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, old)

            env_states = jax.tree.map(select, reset_states, next_states)
            transition = Transition(
                obs=obs_n,
                desc=desc,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=next_states.reward,
                done=next_states.done,
                truncated=next_states.truncated,
            )
            info = (finished_return, finished_mask, tracking_error, env_states.obs)
            return (env_states, rng), (transition, info)

        (env_states, rng), (transitions, infos) = jax.lax.scan(
            rollout_step,
            (carry["env_states"], carry["rng"]),
            None,
            length=self.rollout_length,
        )

        # Bootstrap value for the final observation.
        rng, k_last = jax.random.split(rng)
        _, _, _, last_value = self._policy_step(
            params, normalizer, env_states, k_last
        )

        # GAE. Truncated episodes bootstrap; terminal failures do not.
        gamma = float(ppo["gamma"])
        lam = float(ppo["gae_lambda"])

        def gae_step(next_vals, tr):
            next_adv, next_value = next_vals
            nonterminal = 1.0 - tr.done
            bootstrap = jnp.maximum(nonterminal, tr.truncated)
            delta = tr.reward + gamma * next_value * bootstrap - tr.value
            adv = delta + gamma * lam * nonterminal * next_adv
            return (adv, tr.value), adv

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros(self.num_envs), last_value), transitions, reverse=True
        )
        targets = advantages + transitions.value

        # Flatten (T, N) -> (T*N).
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        batch = jax.tree.map(flatten, (transitions, advantages, targets))

        num_minibatches = int(ppo["num_minibatches"])
        batch_size = self.rollout_length * self.num_envs
        mb_size = batch_size // num_minibatches

        def loss_fn(p, mb_transitions, mb_adv, mb_targets):
            obs = mb_transitions.obs
            mean, log_std = self.actor.apply(
                p["actor"], obs.o_g, obs.o_j, obs.o_f, mb_transitions.desc
            )
            log_prob = gaussian_log_prob(mean, log_std, mb_transitions.action)
            ratio = jnp.exp(log_prob - mb_transitions.log_prob)
            adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            clip_eps = float(ppo["clip_epsilon"])
            pg_loss = -jnp.mean(
                jnp.minimum(
                    ratio * adv,
                    jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
                )
            )
            values = self.critic.apply(
                p["critic"], obs.o_g, obs.o_j, obs.o_f, mb_transitions.desc
            )
            v_loss = 0.5 * jnp.mean(
                jnp.square(values - mb_targets[..., None])
            )
            entropy = gaussian_entropy(log_std)
            total = (
                pg_loss
                + float(ppo["value_coef"]) * v_loss
                - float(ppo["entropy_coef"]) * entropy
            )
            return total, (pg_loss, v_loss, entropy)

        grad_fn = jax.grad(loss_fn, has_aux=True)

        def epoch(update_carry, _):
            p, o_state, rng = update_carry
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, batch_size)

            def minibatch(mb_carry, idx):
                p, o_state = mb_carry
                take = jax.lax.dynamic_slice_in_dim(perm, idx * mb_size, mb_size)
                mb_t, mb_a, mb_g = jax.tree.map(lambda x: x[take], batch)
                grads, aux = grad_fn(p, mb_t, mb_a, mb_g)
                updates, o_state = self.tx.update(grads, o_state, p)
                p = optax.apply_updates(p, updates)
                return (p, o_state), aux

            (p, o_state), aux = jax.lax.scan(
                minibatch, (p, o_state), jnp.arange(num_minibatches)
            )
            return (p, o_state, rng), aux

        (params, opt_state, rng), aux = jax.lax.scan(
            epoch, (params, opt_state, rng), None, length=int(ppo["update_epochs"])
        )
        pg_loss, v_loss, entropy = jax.tree.map(jnp.mean, aux)

        finished_return, finished_mask, tracking_error, raw_obs = infos
        normalizer = normalizer.update(raw_obs)
        ep_count = jnp.maximum(finished_mask.sum(), 1.0)
        metrics = {
            "mean_episode_return": (finished_return * finished_mask).sum() / ep_count,
            "episodes_finished": finished_mask.sum(),
            "mean_reward": transitions.reward.mean(),
            "tracking_error": tracking_error.mean(),
            "pg_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy,
            "mean_value": transitions.value.mean(),
        }
        new_carry = {
            "rng": rng,
            "env_states": env_states,
            "params": params,
            "opt_state": opt_state,
            "normalizer": normalizer,
        }
        return new_carry, metrics

    # ------------------------------------------------------------------ #

    def train(self, seed: int = 0, num_iterations: int | None = None, log_fn=None):
        rng = jax.random.PRNGKey(seed)
        carry = self.init_state(rng)
        total = num_iterations or self.num_iterations
        log_cfg = self.cfg["logging"]
        history = []
        start = time.time()
        for it in range(total):
            carry, metrics = self._train_iter(carry, self.curriculum.c)
            metrics = {k: float(v) for k, v in metrics.items()}
            metrics["curriculum_c"] = self.curriculum.c
            metrics["env_steps"] = (it + 1) * self.steps_per_iter
            self.curriculum.record(metrics["mean_episode_return"])
            expanded = self.curriculum.maybe_expand(it + 1)
            if expanded:
                print(f"[curriculum] expanded design box to c={self.curriculum.c:.2f}")
            history.append(metrics)
            if it % int(log_cfg["log_every_iters"]) == 0 or it == total - 1:
                sps = metrics["env_steps"] / (time.time() - start)
                print(
                    f"iter {it:5d} | steps {metrics['env_steps']:.2e} | "
                    f"ep_return {metrics['mean_episode_return']:8.2f} | "
                    f"reward {metrics['mean_reward']:7.4f} | "
                    f"track_err {metrics['tracking_error']:5.2f} | "
                    f"c {self.curriculum.c:.2f} | {sps:,.0f} sps"
                )
            if log_fn is not None:
                log_fn(it, metrics)
            if it > 0 and it % int(log_cfg["save_every_iters"]) == 0:
                self.save(carry, log_cfg["out_dir"])
        self.save(carry, log_cfg["out_dir"])
        return carry, history

    def save(self, carry, out_dir: str, name: str = "checkpoint.pkl"):
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "params": jax.device_get(carry["params"]),
            "normalizer": jax.device_get(carry["normalizer"]),
            "curriculum_c": self.curriculum.c,
            "config": self.cfg,
        }
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[checkpoint] saved {path}")
        return path
