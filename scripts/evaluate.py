#!/usr/bin/env python
"""Evaluate a design (nominal, random, or a saved f*) with real rollouts.

Usage:
    python scripts/evaluate.py --checkpoint .../checkpoint.pkl              # nominal
    python scripts/evaluate.py --design .../design_nominal.npz --which f_opt
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.utils.evaluation import evaluate_design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/g1_results/checkpoint.pkl")
    parser.add_argument("--design-space", default="configs/g1_design_space.yaml")
    parser.add_argument("--design", default=None, help=".npz with f_init/f_opt")
    parser.add_argument("--which", default="f_opt", choices=["f_init", "f_opt"])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from scripts.design_search import build, load_checkpoint

    ckpt = load_checkpoint(args.checkpoint)
    env, actor, critic, cfg = build(ckpt, args.design_space)

    if args.design:
        f = jnp.asarray(np.load(args.design)[args.which])
        label = f"{os.path.basename(args.design)}:{args.which}"
    else:
        f = jnp.zeros(env.design_map.dim)
        label = "nominal G1 (f = 0)"

    metrics = evaluate_design(
        env, actor, ckpt["params"]["actor"], ckpt["normalizer"], f,
        jax.random.PRNGKey(args.seed),
        num_envs=args.num_envs, episode_length=args.episode_length,
    )
    print(f"design: {label}")
    for key, value in metrics.items():
        print(f"  {key:22s} {value:10.4f}")


if __name__ == "__main__":
    main()
