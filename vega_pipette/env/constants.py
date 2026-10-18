"""Joint names, asset locations and scene geometry of the task (SI units)."""
from __future__ import annotations

from ..paths import ASSETS

ROBOT_USD = ASSETS / "robot" / "wuji_wrist_contact_overlay.usda"
"""Vega-U + Wuji Hand 2 with contact reporting on the right wrist (layers over generated/vega_u_wuji_hand2.usda)."""
PIPETTE_USD = ASSETS / "pipette_v5" / "pipette_calibrated.usda"
PROPS_DIR = ASSETS / "props"
LAB_PROPS_DIR = PROPS_DIR / "piston"
PACKING_TABLE_USD = PROPS_DIR / "PackingTable_NoProps.usda"
"""The packing table (tools/fetch_packing_table.py) with its decorative containers deactivated."""

LEFT_ARM_JOINT_NAMES = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"R_arm_j{i}" for i in range(1, 8)]
_HAND_SUFFIXES = [
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_finger_mcp_flex", "index_finger_mcp_abd", "index_finger_pip", "index_finger_dip",
    "middle_finger_mcp_flex", "middle_finger_mcp_abd", "middle_finger_pip", "middle_finger_dip",
    "ring_finger_mcp_flex", "ring_finger_mcp_abd", "ring_finger_pip", "ring_finger_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
]
LEFT_HAND_JOINT_NAMES = [f"l_{name}" for name in _HAND_SUFFIXES]
RIGHT_HAND_JOINT_NAMES = [f"r_{name}" for name in _HAND_SUFFIXES]
ACTION_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
"""The 54 joints of the policy's state and action, in their order."""
HELD_JOINT_NAMES = ("Lift", "torso_flip", "head_j1", "head_j2", "head_j3")
"""Joints pinned at their defaults during a run."""

SURFACE_Z = 0.80
"""Height of the work table's top [m]; the robot stands at the origin facing +y, the table at y = 0.55."""
EGO_CAMERA_HEIGHT, EGO_CAMERA_WIDTH = 240, 424

assert len(ACTION_JOINT_NAMES) == 54
