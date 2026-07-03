"""Plots and renders for training curves, VGDS convergence and design deltas."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(history: list, out_path: str):
    plt = _plt()
    steps = [h["env_steps"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("mean_episode_return", "Episode return"),
        ("tracking_error", "Velocity tracking error [m/s]"),
        ("value_loss", "Value loss"),
        ("curriculum_c", "Design curriculum c"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(steps, [h[key] for h in history])
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[viz] wrote {out_path}")


def plot_vgds_convergence(history: dict, out_path: str):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["value"])
    axes[0].set_title("Mean ensemble value V̄")
    axes[0].set_xlabel("VGDS iteration")
    axes[1].plot(history["grad_norm"])
    axes[1].set_title("||∇f V̄||")
    axes[1].set_xlabel("VGDS iteration")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[viz] wrote {out_path}")


def plot_design_changes(f_opt, f_ref, labels: list, out_path: str, top_k: int = 40):
    """Bar chart of the largest normalized design changes, colored by group."""
    plt = _plt()
    delta = np.asarray(f_opt) - np.asarray(f_ref)
    order = np.argsort(-np.abs(delta))[:top_k]
    colors = {"joint": "#4477aa", "link": "#ee6677", "actuator": "#228833",
              "geometry": "#ccbb44"}
    fig, ax = plt.subplots(figsize=(9, 0.28 * top_k + 1.5))
    ys = np.arange(len(order))
    ax.barh(
        ys,
        delta[order],
        color=[colors.get(labels[i].split("/")[0], "gray") for i in order],
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[i] for i in order], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("normalized design change (f* - f_ref)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[viz] wrote {out_path}")


def group_design_changes(f_opt, f_ref, labels: list) -> dict:
    """Mean |change| grouped by parameter category (for text summaries)."""
    delta = np.abs(np.asarray(f_opt) - np.asarray(f_ref))
    groups = defaultdict(list)
    for value, label in zip(delta, labels):
        parts = label.split("/")
        groups[f"{parts[0]}/{parts[-1].split('_')[0]}"].append(value)
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def render_design(env, f, out_path: str, width: int = 640, height: int = 480):
    """Offscreen render of a design variant (best effort, needs OpenGL)."""
    import jax
    import mujoco
    import numpy as np

    try:
        import copy

        model_updates, _, _ = jax.device_get(env.design_map(f))
        model = copy.deepcopy(env.model)
        for field in ("body_mass", "body_ipos", "body_pos", "geom_size", "geom_pos"):
            getattr(model, field)[:] = np.asarray(model_updates[field])
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(env.default_qpos)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data)
        import PIL.Image
        PIL.Image.fromarray(renderer.render()).save(out_path)
        renderer.close()
        print(f"[viz] wrote {out_path}")
    except Exception as exc:  # rendering is optional (headless machines)
        print(f"[viz] render skipped: {exc}")
