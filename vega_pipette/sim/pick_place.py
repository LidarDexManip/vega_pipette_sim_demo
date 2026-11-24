"""One simulated tube pick-and-place scene (env/pick_place.py): Session's stepping and cameras, with this task's poses.

Import after AppLauncher(...) has started Isaac Sim.
"""
from __future__ import annotations

import numpy as np

from ..env import pick_place as pp
from .session import Session

VIEW_EYE, VIEW_LOOKAT = (-1.30, 1.45, 1.55), (-0.25, 0.40, 0.95)
"""Window viewport: in front of the robot and to its left, looking at the rack (world frame, m)."""
POLICY_VIEWS = ("ego_view", "left_palm_view")
"""The cameras the policy reads."""


def _np(x: object) -> np.ndarray:
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(float)


class PickPlaceSession(Session):
    """The pick-and-place scene of one layout {rack_dxy, hole_a, hole_b}, with one environment."""

    def __init__(self, layout: dict, device: str, window: bool = False, cam_scale: int = 1,
                 side_camera: bool = True) -> None:
        self._open(pp.build_pick_place_cfg(layout, device, side_camera=side_camera), pp.CAMERAS, window, cam_scale,
                   (VIEW_EYE, VIEW_LOOKAT))
        scene = self.control.base.scene
        self.tube, self.rack = scene["tube"], scene["centrifuge_tube"]
        self._wrist = self.robot.find_bodies("l_wrist")[0][0]

    def poses(self) -> dict[str, np.ndarray]:
        """Tube, rack, left wrist and robot root poses of this step (positions in m, quaternions xyzw)."""
        return {"tube_pos": _np(self.tube.data.root_pos_w[0]), "tube_quat": _np(self.tube.data.root_quat_w[0]),
                "rack_pos": _np(self.rack.data.root_pos_w[0]), "rack_quat": _np(self.rack.data.root_quat_w[0]),
                "lw_pos": _np(self.robot.data.body_pos_w[0, self._wrist]),
                "lw_quat": _np(self.robot.data.body_quat_w[0, self._wrist]),
                "root_pos": _np(self.robot.data.root_pos_w[0]), "root_quat": _np(self.robot.data.root_quat_w[0])}
