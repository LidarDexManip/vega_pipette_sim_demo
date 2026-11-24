"""Forward kinematics of the robot in numpy, from its kinematic description
(assets/robot/vega_u_wuji_hand2_kinematics.urdf).

left_ee(q7)       the left end effector L_ee in the robot's root frame from the 7 left-arm joints (the lift and the
                  torso joint at the values the scene holds them at);
hand_links(q20)   the left hand's links in the left wrist's frame from its 20 joints.
Joint values are the 54-D state's: the left arm at [0:7], the left hand at [7:27] (env/constants.py).
"""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .. import paths
from ..env.constants import ACTION_JOINT_NAMES

URDF = paths.ASSETS / "robot" / "vega_u_wuji_hand2_kinematics.urdf"
HELD = {"Lift": 0.15, "torso_flip": 0.0}
"""The joints the scene holds still, at their values (m, rad)."""
LEFT_ARM = ACTION_JOINT_NAMES[0:7]
LEFT_HAND = ACTION_JOINT_NAMES[7:27]


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """The rotation of a URDF origin's roll, pitch and yaw."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(a: np.ndarray, t: float) -> np.ndarray:
    """The rotation by angle t about axis a."""
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


Joint = tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None]
"""name, type, origin position, origin rotation, axis."""


@functools.lru_cache(maxsize=None)
def chain(link: str, base: str, urdf: Path = URDF) -> tuple[Joint, ...]:
    """The joints from `base` down to `link`, in order."""
    root = ET.parse(urdf).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    joints = []
    while link != base:
        j = by_child[link]
        joints.append(j)
        link = j.find("parent").get("link")
    out = []
    for j in reversed(joints):
        o, a = j.find("origin"), j.find("axis")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
        out.append((j.get("name"), j.get("type"), xyz, rpy_matrix(*rpy), axis))
    return tuple(out)


def pose(joints: tuple[Joint, ...], values: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and position of the end of a joint chain in the frame of its start."""
    R, p = np.eye(3), np.zeros(3)
    for name, kind, xyz, Ro, axis in joints:
        p = p + R @ xyz
        R = R @ Ro
        if kind == "fixed":
            continue
        v = values[name]
        if kind == "prismatic":
            p = p + R @ (axis * v)
        else:
            R = R @ axis_angle(axis, v)
    return R, p


def left_ee(q7: ArrayLike) -> np.ndarray:
    """L_ee's position (m) in the robot's root frame."""
    q7 = np.asarray(q7, np.float64)
    return pose(chain("L_ee", "base_link"), {**HELD, **dict(zip(LEFT_ARM, q7))})[1]


def hand_links(q20: ArrayLike, links: tuple[str, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(rotation, position) of each of `links` in the left wrist's (l_wrist) frame."""
    values = dict(zip(LEFT_HAND, np.asarray(q20, np.float64)))
    return [pose(chain(link, "l_wrist"), values) for link in links]
