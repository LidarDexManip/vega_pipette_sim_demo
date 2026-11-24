# Vega-U left end effector kinematics for openpi, by LIDAR (Apache License 2.0).
"""Forward kinematics of the Vega-U's left end effector (L_ee) in numpy, for the tube pick-and-place policy's
"Rel: dx dy" text (vega_policy.pocket_rel_text).

The joint chain is read once from the robot's kinematic description, a URDF the server names with `use_urdf`
(policy_server/serve.py: the repository's assets/robot/vega_u_wuji_hand2_kinematics.urdf). The lift and the torso
joint are held at the values the simulated scene holds them at. The simulation runner computes the same chain in
vega_pipette/control/kinematics.py.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import numpy as np

HELD = {"Lift": 0.15, "torso_flip": 0.0}
_urdf: str | None = None


def use_urdf(path: str) -> None:
    """The URDF that `left_ee` reads."""
    global _urdf
    _urdf = str(path)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, t):
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


@functools.lru_cache(maxsize=2)
def _chain(urdf: str):
    root = ET.parse(urdf).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    link, chain = "L_ee", []
    while link in by_child:
        j = by_child[link]
        chain.append(j)
        link = j.find("parent").get("link")
    out = []
    for j in reversed(chain):
        o, a = j.find("origin"), j.find("axis")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
        out.append((j.get("name"), j.get("type"), xyz, _rpy(*rpy), axis))
    return tuple(out)


def left_ee(q7) -> np.ndarray:
    """L_ee position [m] in the robot's root frame for the 7 left-arm joints (54-D state[:7])."""
    if _urdf is None:
        raise RuntimeError("vega_fk: no URDF; call use_urdf(path) first")
    q7 = np.asarray(q7, dtype=np.float64)
    R, p = np.eye(3), np.zeros(3)
    for name, kind, xyz, Ro, axis in _chain(_urdf):
        p = p + R @ xyz
        R = R @ Ro
        if kind == "fixed":
            continue
        v = HELD[name] if name in HELD else q7[int(name.rsplit("j", 1)[1]) - 1]
        if kind == "prismatic":
            p = p + R @ (axis * v)
        else:
            R = R @ _axis_angle(axis, v)
    return p
