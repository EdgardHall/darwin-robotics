from setuptools import find_packages, setup

setup(
    name="darwin",
    version="0.1.0",
    description=(
        "Shape Your Body on the Unitree G1: multi-embodiment PPO (URMA) "
        "+ Value-Gradient Design Search (VGDS) in MuJoCo MJX"
    ),
    packages=find_packages(include=["darwin", "darwin.*"]),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.30",
        "mujoco>=3.2.0",
        "mujoco-mjx>=3.2.0",
        "flax>=0.8.0",
        "optax>=0.2.0",
        "numpy",
        "PyYAML",
    ],
)
