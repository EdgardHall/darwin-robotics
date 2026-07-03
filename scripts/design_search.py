#!/usr/bin/env python
"""Run Value-Gradient Design Search with a trained checkpoint.

Usage:
    python scripts/design_search.py --checkpoint experiments/g1_results/checkpoint.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.design_search.state_bank import collect_state_bank
from darwin.design_search.vgds import make_value_fn, vgds
from darwin.envs.g1_environment import G1Env
from darwin.models.urma_critic import URMACritic
from darwin.models.urma_policy import URMAActor
from darwin.utils.evaluation import evaluate_design
from darwin.utils.visualization import (
    group_design_changes,
    plot_design_changes,
    plot_vgds_convergence,
    render_design,
)


def load_checkpoint(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build(ckpt, design_space_path):
    cfg = ckpt["config"]
    with open(design_space_path) as fh:
        bounds = yaml.safe_load(fh)
    env = G1Env(cfg, bounds)
    net = cfg["network"]
    actor = URMAActor(
        joint_encoder_hidden=net["joint_encoder_hidden"],
        attention_key_hidden=net["attention_key_hidden"],
        core_hidden=net["core_hidden"],
        latent_dim=net["latent_dim"],
        init_log_std=net["init_log_std"],
    )
    critic = URMACritic(
        joint_encoder_hidden=net["joint_encoder_hidden"],
        attention_key_hidden=net["attention_key_hidden"],
        core_hidden=net["core_hidden"],
        latent_dim=net["latent_dim"],
        num_heads=net["num_critic_heads"],
    )
    return env, actor, critic, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/g1_results/checkpoint.pkl")
    parser.add_argument("--config", default="configs/design_search.yaml")
    parser.add_argument("--design-space", default="configs/g1_design_space.yaml")
    parser.add_argument("--out-dir", default="experiments/g1_results/design_search")
    parser.add_argument("--num-initial-designs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true",
                        help="also evaluate designs with real rollouts")
    args = parser.parse_args()

    with open(args.config) as fh:
        ds_cfg = yaml.safe_load(fh)
    ckpt = load_checkpoint(args.checkpoint)
    env, actor, critic, cfg = build(ckpt, args.design_space)
    params = ckpt["params"]
    normalizer = ckpt["normalizer"]
    os.makedirs(args.out_dir, exist_ok=True)

    rng = jax.random.PRNGKey(int(ds_cfg.get("seed", 0)))
    dim = env.design_map.dim
    f_ref = jnp.zeros(dim)

    # ---- Phase 5.1: state bank --------------------------------------------
    bank_cfg = ds_cfg["state_bank"]
    rng, k_bank = jax.random.split(rng)
    print(f"[state bank] collecting {bank_cfg['num_states']} states "
          f"from {bank_cfg['num_designs']} designs...")
    state_bank = collect_state_bank(
        env, actor, params["actor"], normalizer, k_bank,
        num_states=int(bank_cfg["num_states"]),
        num_designs=int(bank_cfg["num_designs"]),
        rollout_length=int(bank_cfg["rollout_length"]),
        design_range=float(bank_cfg["design_range"]),
    )
    print(f"[state bank] collected {jax.tree.leaves(state_bank)[0].shape[0]} states")

    # ---- Phase 5.3: VGDS from several starting designs ---------------------
    v_cfg = ds_cfg["vgds"]
    iterations = args.iterations or int(v_cfg["iterations"])
    n_starts = args.num_initial_designs or int(
        ds_cfg["evaluation"]["num_initial_designs"]
    )
    value_fn = jax.jit(make_value_fn(critic, params["critic"], normalizer,
                                     env.design_map))
    bank_all = state_bank

    results = []
    for i in range(n_starts):
        rng, k_init, k_run, k_eval = jax.random.split(rng, 4)
        f_init = (
            jnp.zeros(dim) if i == 0
            else jax.random.uniform(k_init, (dim,), minval=-1.0, maxval=1.0)
        )
        f_opt, history = vgds(
            critic, params["critic"], normalizer, env.design_map,
            state_bank, f_init, f_ref, k_run,
            iterations=iterations,
            batch_size=int(v_cfg["batch_size"]),
            learning_rate=float(v_cfg["learning_rate"]),
            trust_region_lambda=float(v_cfg["trust_region_lambda"]),
            delta_max=float(v_cfg["delta_max"]),
            f_clip=float(v_cfg["f_clip"]),
        )
        entry = {
            "start": "nominal" if i == 0 else f"random_{i}",
            "value_init": float(value_fn(f_init, bank_all)),
            "value_opt": float(value_fn(f_opt, bank_all)),
        }
        if args.evaluate:
            e_cfg = ds_cfg["evaluation"]
            metrics_init = evaluate_design(
                env, actor, params["actor"], normalizer, f_init, k_eval,
                num_envs=int(e_cfg["num_eval_envs"]))
            metrics_opt = evaluate_design(
                env, actor, params["actor"], normalizer, f_opt, k_eval,
                num_envs=int(e_cfg["num_eval_envs"]))
            entry["rollout_init"] = metrics_init
            entry["rollout_opt"] = metrics_opt
            entry["delta_return"] = (
                metrics_opt["mean_return"] - metrics_init["mean_return"]
            )
        results.append(entry)
        np.savez(
            os.path.join(args.out_dir, f"design_{entry['start']}.npz"),
            f_init=np.asarray(f_init), f_opt=np.asarray(f_opt),
            value_history=np.asarray(history["value"]),
        )
        print(f"[vgds] {entry['start']}: V̄ {entry['value_init']:.3f} -> "
              f"{entry['value_opt']:.3f}"
              + (f" | rollout ΔR = {entry['delta_return']:+.2f}"
                 if args.evaluate else ""))
        if i == 0:
            try:
                plot_vgds_convergence(
                    history, os.path.join(args.out_dir, "vgds_convergence.png"))
                plot_design_changes(
                    np.asarray(f_opt), np.asarray(f_ref),
                    env.design_map.labels,
                    os.path.join(args.out_dir, "design_changes.png"))
            except ImportError:
                pass
            print("[analysis] mean |Δf| by group:")
            for key, value in group_design_changes(
                    np.asarray(f_opt), np.asarray(f_ref),
                    env.design_map.labels).items():
                print(f"    {key:30s} {value:.4f}")
            render_design(env, f_opt, os.path.join(args.out_dir, "design_opt.png"))

    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] results in {args.out_dir}")


if __name__ == "__main__":
    main()
