#!/usr/bin/env python
"""Train the multi-embodiment URMA policy on randomized G1 morphologies.

Usage:
    python scripts/train.py --config configs/training.yaml
    python scripts/train.py --num-envs 8 --iterations 5        # CPU smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin.envs.g1_environment import G1Env
from darwin.training.ppo_trainer import PPOTrainer
from darwin.utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--design-space", default="configs/g1_design_space.yaml")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--fixed-design", action="store_true",
                        help="disable morphology randomization (c=0 smoke test)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(args.design_space) as fh:
        bounds = yaml.safe_load(fh)

    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    if args.rollout_length:
        cfg["ppo"]["rollout_length"] = args.rollout_length
    if args.episode_length:
        cfg["env"]["episode_length"] = args.episode_length
    if args.fixed_design:
        cfg["curriculum"]["c_init"] = 0.0
        cfg["curriculum"]["c_final"] = 0.0
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))

    env = G1Env(cfg, bounds)
    print(f"design space dimension: {env.design_map.dim}")
    trainer = PPOTrainer(env, cfg)
    print(
        f"training: {trainer.num_iterations if args.iterations is None else args.iterations}"
        f" iterations x {trainer.steps_per_iter} env steps"
    )

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    def log_fn(it, metrics):
        import json

        metrics_file.write(json.dumps({"iter": it, **metrics}) + "\n")
        metrics_file.flush()

    carry, history = trainer.train(
        seed=seed, num_iterations=args.iterations, log_fn=log_fn
    )
    metrics_file.close()
    try:
        plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))
    except ImportError:
        print("[viz] matplotlib unavailable; skipped curves")


if __name__ == "__main__":
    main()
