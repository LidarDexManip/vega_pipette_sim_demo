"""The environment config of the tube pick-and-place task: env/task.py's lab scene, set up for the left hand alone.

build_pick_place_cfg(layout, device) starts from the pipette task's config and changes:
  - props: the pipette, its stand, the mini centrifuge and the pot are not in the scene; the rack is the generated one
    (env/objects/tube_rack.py, every pocket open), placed RACK_CENTRE + layout rack_dxy; the bevelled tube
    (env/objects/tube.py) stands in pocket hole_a;
  - robot: the head turned down toward the rack (HEAD_POSE), the right arm resting at the side (RIGHT_ARM_REST; it
    holds that pose for the whole run);
  - cameras: the head camera at the real robot's mount (its optical frame as is) rendered at the policy's 224 x 126,
    the palm-side left wrist camera at 224 x 224, and optionally a side view for people.
Quaternions are xyzw.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_matrix

from ..progress import rack
from ..progress.geometry import quat_matrix
from .objects.tube import bevel_tube
from .objects.tube_rack import TubeRackCfg
from .scene import (
    TUBE_COM_LOCAL_Z,
    TUBE_HEIGHT,
    TUBE_MASS,
    TUBE_RADIUS,
    TUBE_SPAWN_Z,
    WRIST_PRIM_PATH,
    look_at,
    wrap_spawn,
)
from .task import LEFT_HAND_MU, TUBE_RACK_POS, TaskEnvCfg

RACK_CENTRE = (TUBE_RACK_POS[0] + 0.10, TUBE_RACK_POS[1] + 0.08)
"""The rack's nominal place (world x, y): 10 cm right of and 8 cm farther than the pipette task's rack."""
RACK_FRICTION = 0.3
"""Friction of the rack's collider; the tube's material (LEFT_HAND_MU, combine "min") makes it the tube-rack value."""
HEAD_POSE = {"head_j1": 1.46608, "head_j2": 0.0, "head_j3": 0.95}
RIGHT_ARM_REST = (-2.2, -0.40, 0.0, -0.9, 0.0, 0.0, 0.0)
EGO_HEIGHT, EGO_WIDTH = 126, 224
PALM_CAM_POS = (0.007, 0.0723, -0.0587)                    # in the left wrist frame
PALM_CAM_ROT = (0.36082074947404175, 0.6917028406741591, 0.5546672700840087, 0.2893091537608909)   # opengl
PALM_CAM_FOCAL, PALM_CAM_APERTURE, PALM_CAM_SIZE = 18.148, 20.955, 224
SIDE_CAM_EYE = (-0.80, 1.00, 1.30)
SIDE_CAM_TARGET = (-0.30, 0.44, 0.90)

CAMERAS = {"ego_view": "head_camera", "left_palm_view": "left_palm_camera", "side_view": "side_camera"}
"""Camera view name -> scene attribute."""


def build_pick_place_cfg(layout: dict, device: str, side_camera: bool = True) -> TaskEnvCfg:
    """The env config for one layout {rack_dxy, hole_a, hole_b} (sim/layouts.py); without the side camera if not
    `side_camera` (the policy does not see it)."""
    cfg = TaskEnvCfg()
    cfg.sim.device = device
    scene = cfg.scene
    scene.pot = scene.pipette_stand = scene.pipette_v5 = scene.mini_centrifuge = None
    cfg.events.pipette_friction = None

    robot = scene.robot
    robot.init_state.joint_pos = {**robot.init_state.joint_pos, **HEAD_POSE,
                                  **{f"R_arm_j{k + 1}": v for k, v in enumerate(RIGHT_ARM_REST)}}

    # the rack (in the pipette task's rack slot, so the scene keeps its creation order) and the tube in pocket hole_a
    rack_xy = (RACK_CENTRE[0] + layout["rack_dxy"][0], RACK_CENTRE[1] + layout["rack_dxy"][1])
    scene.centrifuge_tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TubeRack", spawn=TubeRackCfg(friction=RACK_FRICTION),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(rack_xy[0], rack_xy[1], TUBE_RACK_POS[2]),
                                                  rot=(0.0, 0.0, 0.0, 1.0)))
    hole = rack.hole_xy(layout["hole_a"])
    tube_spawn = sim_utils.CylinderCfg(
        radius=TUBE_RADIUS, height=TUBE_HEIGHT, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=TUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.15), metallic=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min", restitution_combine_mode="min",
            static_friction=LEFT_HAND_MU, dynamic_friction=LEFT_HAND_MU, restitution=0.0))

    def tube_com(prim: object, _path: str) -> None:
        from pxr import Gf, UsdPhysics
        UsdPhysics.MassAPI.Apply(prim).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, TUBE_COM_LOCAL_Z))

    wrap_spawn(tube_spawn, lambda prim, _path: bevel_tube(prim))
    wrap_spawn(tube_spawn, tube_com)
    scene.tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube", spawn=tube_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(rack_xy[0] + float(hole[0]), rack_xy[1] + float(hole[1]), TUBE_SPAWN_Z), rot=(0.0, 0.0, 0.0, 1.0)))
    table = scene.tabletop_collider.spawn
    table.physics_material = table.physics_material.replace(friction_combine_mode="min")

    # cameras: the head camera with its optical frame as mounted, the palm camera, the side view (optional)
    scene.head_camera = scene.head_camera.replace(
        height=EGO_HEIGHT, width=EGO_WIDTH,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"))
    rot = quat_from_matrix(torch.tensor(quat_matrix(PALM_CAM_ROT)[None], dtype=torch.float32))[0]
    scene.left_palm_camera = CameraCfg(
        prim_path=f"{WRIST_PRIM_PATH['L']}/PalmCamera", update_period=0.0, height=PALM_CAM_SIZE, width=PALM_CAM_SIZE,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=PALM_CAM_FOCAL, horizontal_aperture=PALM_CAM_APERTURE, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=PALM_CAM_POS, rot=tuple(float(v) for v in rot), convention="opengl"))
    if side_camera:
        scene.side_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera", update_period=0.0, height=360, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=SIDE_CAM_EYE, rot=look_at(SIDE_CAM_EYE, SIDE_CAM_TARGET),
                                       convention="opengl"))
    return cfg
