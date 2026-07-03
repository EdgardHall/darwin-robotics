# DARWINROBOTICS — Shape Your Body on the Unitree G1

Reproduction of the **VGDS** method (*Value-Gradient Design Search*, "Shape
Your Body", Bohlinger & Peters, 2025) focused exclusively on the **Unitree G1
humanoid** (29 DOF).

The pipeline:

1. **Multi-embodiment training** — a URMA policy/critic is trained with PPO in
   MuJoCo MJX across thousands of parallel environments, each simulating a
   *different* randomized G1 morphology (masses, inertias, joint limits,
   PD gains, CoM offsets, foot geometry, leg lengths — 578 design dimensions).
   Every episode reset draws a fresh design `f ~ U([-c, c]^578)` where `c`
   grows with a curriculum.
2. **Value-Gradient Design Search** — after training, the frozen critic
   ensemble becomes a differentiable surrogate of design quality. VGDS
   performs Adam ascent on `f` through `∇f V̄(s, Φ(f))`, evaluated on a bank
   of diverse states, with a trust-region penalty. The simulator is never
   stepped during the search.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

G1 assets (already vendored in `assets/g1/`) come from
[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
(`g1_29dof.xml` + meshes, BSD-3-Clause).

## Usage

```bash
# Smoke test on CPU (sanity check, ~1 min)
python scripts/train.py --num-envs 8 --iterations 3 --rollout-length 16 --episode-length 100

# Fixed-design smoke test (verification plan #2: no randomization)
python scripts/train.py --fixed-design --num-envs 1024 --iterations 250

# Full multi-embodiment training (~1e9 steps; needs a GPU, e.g. A100)
python scripts/train.py --config configs/training.yaml

# Design search with the trained checkpoint (+ real-rollout evaluation)
python scripts/design_search.py --checkpoint experiments/g1_results/checkpoint.pkl --evaluate

# Evaluate one design with rollouts
python scripts/evaluate.py --design experiments/g1_results/design_search/design_nominal.npz --which f_opt
```

Tests (design-map roundtrip, network shapes, env stepping, VGDS gradients):

```bash
for t in tests/test_*.py; do .venv/bin/python "$t"; done
```

## Architecture notes

- **`darwin/models/design_map.py`** — `Phi(f)`: pure-JAX mapping from
  normalized `f ∈ [-1,1]^578` to MJX model fields (`body_mass`,
  `body_inertia`, `body_ipos`, `jnt_range`, `jnt_pos`, `dof_damping`,
  `dof_armature`, `jnt_stiffness`, `geom_size/pos`, `body_pos`), PD/actuation
  parameters, and URMA description vectors. Hard clamps keep every design
  physical (positive masses/inertias, ordered joint limits).
- **`darwin/models/urma_critic.py`** — *Direct-Design* URMA critic: the
  per-joint encoder consumes `concat(o_j, d_j)` (descriptions flow through
  attention **values**, not just keys) and feeds an ensemble of K=5 value
  heads; `V̄` is the head mean. Both give stronger, better-conditioned
  `∇f V̄` for the search.
- **`darwin/envs/g1_environment.py`** — functional MJX env, `jax.vmap`-ed by
  the trainer; each env carries its own model-field overrides
  (`mjx.Model.tree_replace`) so 4096 morphologies step in one compiled call.
  Rudin-style velocity-tracking rewards, PD control at 50 Hz over 2 ms physics.
- **`darwin/design_search/vgds.py`** — the search itself:
  `f ← clip(f + clip(Adam(∇f J), ±0.05), ±1)` with
  `J(f) = mean_s V̄(s, Φ(f)) − λ‖f − f_ref‖²/D`.

## Known deviations from the plan

- PPO is a self-contained JAX/Flax implementation *modeled on* RL-X rather
  than importing it (avoids a heavy dependency; hyperparameters match).
- Link inertia is randomized on the 3 diagonal components (not 6) — arbitrary
  off-diagonal scaling can produce non-physical inertia tensors.
- Self-collision penalty is omitted: mesh self-collisions are disabled for
  MJX (feet-only contacts), and falls are handled by termination.
- Collision geometry is reduced to the 8 foot spheres (standard MJX practice
  for the G1); mesh visuals do not stretch when link lengths change.
