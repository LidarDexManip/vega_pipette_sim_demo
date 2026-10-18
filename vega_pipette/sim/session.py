"""One simulated pipette scene: build it, settle it, read the robot and the cameras, step it.

Import after AppLauncher(...) has started Isaac Sim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from scipy.spatial.transform import Rotation

from ..env import scene as sc
from ..env.constants import ACTION_JOINT_NAMES, HELD_JOINT_NAMES
from ..env.control import Control
from . import look

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

SETTLE_STEPS = 60
"""Steps of the default action, with the lift / torso / head joints pinned, before the policy starts."""

VIEW_EYE, VIEW_LOOKAT = (1.45, 1.75, 1.55), (-0.05, 0.50, 1.00)
"""Window viewport: behind and above the right shoulder, looking at the tube rack (world frame, m)."""

ACTION_HZ = sc.CONTROL_HZ
STATE_GROUPS = sc.STATE_GROUPS


class Session:
    """The scene of one layout, with one environment.

    window: frame the simulator window's viewport. cam_scale: every camera's resolution times this (the field of view
    is kept). side_camera: also render the side view (the policy does not see it; the dashboard and the videos do).
    """

    def __init__(self, layout: dict, device: str, window: bool = False, cam_scale: int = 1,
                 side_camera: bool = True) -> None:
        cfg = sc.build_environment_cfg(layout, device, side_camera=side_camera)
        names = {view: name for view, name in sc.CAMERAS.items() if getattr(cfg.scene, name, None) is not None}
        if cam_scale != 1:
            for name in names.values():
                cam = getattr(cfg.scene, name)
                cam.height, cam.width = cam.height * cam_scale, cam.width * cam_scale
        if window:
            cfg.viewer.eye, cfg.viewer.lookat = VIEW_EYE, VIEW_LOOKAT
        look.install(log=lambda m: None)
        self.control = Control(ManagerBasedRLEnv(cfg=cfg))
        look.apply_carb(log=lambda m: None)
        if window:
            look.show_viewport(VIEW_EYE, VIEW_LOOKAT)
        scene = self.control.base.scene
        self.robot, self.pipette, self.tube = scene["robot"], scene["pipette_v5"], scene["tube"]
        self.stand, self.rack = scene["pipette_stand"], scene["centrifuge_tube"]
        self._cams = {view: scene[name] for view, name in names.items() if view != "ego_view"}
        self.views = tuple(names)
        """The views this scene renders."""
        self._joints = [self.robot.find_joints(n)[0][0] for n in ACTION_JOINT_NAMES]
        self._wrist = {arm: self.robot.find_bodies(sc.WRIST_BODY[arm])[0][0] for arm in sc.ARMS}
        self._thumb_r = self.robot.find_bodies(sc.THUMB_BODY["R"])[0][0]
        self._plunger = self.pipette.find_joints("PlungerJoint")[0][0]
        self.frames: dict[str, np.ndarray] = {}
        """The latest image of every view, uint8 RGB (H, W, 3)."""
        self.default_action = np.zeros(54, np.float32)

    def settle(self) -> None:
        """Reset, pin the lift / torso / head joints and run the default action for SETTLE_STEPS steps."""
        self.control.reset()
        self.default_action = self.control.default_action().copy()
        held, _ = self.robot.find_joints(list(HELD_JOINT_NAMES), preserve_order=True)
        pin = self.robot.data.default_joint_pos[:, held]
        self.robot.write_joint_position_limit_to_sim_index(
            limits=torch.stack([pin - 2.0e-3, pin + 2.0e-3], dim=-1), joint_ids=held)
        self.robot.set_joint_position_target_index(target=pin, joint_ids=held)
        for _ in range(SETTLE_STEPS):
            self.step(self.default_action)
        bad = look.check(len(self.views))
        if bad:
            raise RuntimeError(f"render settings not in place: {bad}")

    def step(self, action: np.ndarray) -> None:
        """Apply one action (54-D absolute joint targets) and read the cameras."""
        obs = self.control.step(action)
        self.frames["ego_view"] = sc.rgb8(obs["rgb"])
        for view, cam in self._cams.items():
            self.frames[view] = sc.rgb8(cam.data.output["rgb"][0].detach().cpu().numpy())

    def state(self) -> np.ndarray:
        """Measured joint positions, 54-D, in the action order."""
        return self.robot.data.joint_pos[0, self._joints].detach().cpu().numpy().astype(np.float32)

    def poses(self) -> dict[str, np.ndarray]:
        """Object, wrist and tip poses of this step (positions in m, quaternions xyzw) and the plunger position."""
        p_pip, r_pip = _root(self.pipette)
        p_tube, r_tube = _root(self.tube)
        p_rw, r_rw = _body(self.robot, self._wrist["R"])
        p_lw, r_lw = _body(self.robot, self._wrist["L"])
        return {"pip_pos": p_pip, "pip_quat": r_pip.as_quat(), "tube_pos": p_tube, "tube_quat": r_tube.as_quat(),
                "rw_pos": p_rw, "lw_pos": p_lw, "tip_pos": p_pip + r_pip.apply(sc.TIP_LOCAL),
                "plunger": float(self.pipette.data.joint_pos[0, self._plunger]),
                "rw_quat": r_rw.as_quat(), "lw_quat": r_lw.as_quat(),
                "rthumb_pos": _body(self.robot, self._thumb_r)[0], "stand_pos": _root(self.stand)[0],
                "rack_pos": _root(self.rack)[0]}

    def close(self) -> None:
        self.control.close()


def _root(asset: RigidObject | Articulation) -> tuple[np.ndarray, Rotation]:
    p = asset.data.root_pos_w[0].detach().cpu().numpy().astype(float)
    q = asset.data.root_quat_w[0].detach().cpu().numpy().astype(float)
    return p, Rotation.from_quat(q)


def _body(asset: Articulation, i: int) -> tuple[np.ndarray, Rotation]:
    p = asset.data.body_pos_w[0, i].detach().cpu().numpy().astype(float)
    q = asset.data.body_quat_w[0, i].detach().cpu().numpy().astype(float)
    return p, Rotation.from_quat(q)
