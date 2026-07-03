#!/usr/bin/env python
"""Convert a URDF into MuJoCo MJCF XML.

The repository already ships the official MJCF from unitree_mujoco
(assets/g1/g1_29dof.xml); this utility exists for regenerating assets from a
URDF source (e.g. unitreerobotics/unitree_ros g1_29dof).

Usage:
    python scripts/convert_urdf.py input.urdf output.xml [--meshdir meshes]
"""

from __future__ import annotations

import argparse

import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    parser.add_argument("output")
    parser.add_argument("--meshdir", default="meshes")
    args = parser.parse_args()

    spec = mujoco.MjSpec.from_file(args.urdf)
    spec.meshdir = args.meshdir
    spec.compile()  # validates joints, inertials, meshes
    with open(args.output, "w") as fh:
        fh.write(spec.to_xml())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
