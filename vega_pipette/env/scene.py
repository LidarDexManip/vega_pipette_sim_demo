"""The environment config of one layout: the task config of env/task.py plus what the layout sets and the policy cameras.

build_environment_cfg(layout, device) places the pipette stand, the pipette on its rail, the tube rack and the tube
by `layout` (sim/layouts.py), and adds the cameras: ego (head, from the task), side, left wrist, right wrist and
right thumb, and the finger-pipette contact sensors.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from scipy.spatial.transform import Rotation

from .constants import PROPS_DIR
from .task import LEFT_HAND_MU, TUBE_RACK_POS, TaskEnvCfg

if TYPE_CHECKING:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from pxr import Usd

CONTROL_HZ = 30.0
ARMS = ("L", "R")
WRIST_BODY = {"L": "l_wrist", "R": "r_wrist"}
THUMB_BODY = {"L": "l_thumb_distal", "R": "r_thumb_distal"}
_ARM = "{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_link/torso_flip_link/arm_center/"
WRIST_PRIM_PATH = {
    "L": _ARM + "L_arm_l1/L_arm_l2/L_arm_l3/L_arm_l4/L_arm_l5/L_arm_l6/L_arm_l7/L_arm_l8/l_wrist",
    "R": _ARM + "R_arm_l1/R_arm_l2/R_arm_l3/R_arm_l4/R_arm_l5/R_arm_l6/R_arm_l7/R_arm_l8/r_wrist",
}
RIGHT_FINGER_DISTAL = (
    ("r_index_finger_distal", "r_index_finger_proximal/r_index_finger_proximal_abd/r_index_finger_middle"),
    ("r_middle_finger_distal", "r_middle_finger_proximal/r_middle_finger_proximal_abd/r_middle_finger_middle"),
    ("r_ring_finger_distal", "r_ring_finger_proximal/r_ring_finger_proximal_abd/r_ring_finger_middle"),
    ("r_pinky_distal", "r_pinky_proximal/r_pinky_proximal_abd/r_pinky_middle"),
)

# ---- tube and rack --------------------------------------------------------------------------------------------------
TUBE_RADIUS = 0.0075
TUBE_HEIGHT = 0.16
TUBE_MASS = 0.08               # weighted base: centre of mass 25 mm above the bottom
TUBE_COM_LOCAL_Z = -0.055
RACK_PLATE_HALF = (0.0453, 0.0843)
RACK_COLLIDER_BANDS = ((0.0, 0.015), (0.0335, 0.0500))   # rack collider bands (z ranges, m)
# pocket grid of the holder mesh: rows at pitch 0.0205 (local x), columns at 0.0230 (local y), odd rows offset by half
RACK_HOLE_BASE = (0.0098, -0.058)
RACK_HOLE_PITCH = (0.0205, 0.0230)
RACK_OPEN_HALF = 0.011
TUBE_SPAWN_SHIFT = (-0.0016, 0.0004)   # tube spawn offset from the pocket centre
TUBE_SPAWN_Z = 0.895
STAND_MASS = 50.0
RACK_YAW_DEG = -90.0
RACK_FORWARD_SHIFT = 0.05
_RAIL_DIR_W = Rotation.from_euler("z", RACK_YAW_DEG, degrees=True).apply(np.array([0.0, 1.0, 0.0]))

# ---- pipette (body frame of assets/pipette_v5/pipette_calibrated.usda, +Z = button) ---------------------------------
TIP_LOCAL = np.array([0.0, 0.0, -0.2436])
PIP_ASSET_MASS = 0.066         # /Pipette/Body mass, kg
PIP_ASSET_COM_Z = -0.01044     # /Pipette/Body centre of mass z, m
PIP_GRIP_MU_S, PIP_GRIP_MU_D, PIP_GRIP_RESTITUTION = 0.8, 0.65, 0.05   # the asset's grip material
# An invisible collider over the lower grip region, so the fingers ride one cross-section when the pipette slides in
# the hand: the body's own seam ring (BodyLowerCollider points[0:16] at z = -0.07134, counter-clockwise from +Z)
# extruded from LOWER_GRIP_Z[0] to LOWER_GRIP_Z[1].
LOWER_GRIP_Z = (-0.1550, -0.07134)
PIP_BODY_SEAM_RING = (
    (0.00945255, 0.0111708), (0.00908889, 0.0125279995), (0.00809535, 0.01352154), (0.00673815, 0.0138852),
    (-0.00673815, 0.0138852), (-0.00809535, 0.01352154), (-0.00908889, 0.0125279995), (-0.00945255, 0.0111708),
    (-0.00945255, -0.0111708), (-0.00908889, -0.0125279995), (-0.00809535, -0.01352154), (-0.00673815, -0.0138852),
    (0.00673815, -0.0138852), (0.00809535, -0.01352154), (0.00908889, -0.0125279995), (0.00945255, -0.0111708),
)
TIP_COLOR = (0.1, 0.75, 0.2)   # opaque green disposable tip
TIP_MESH = "Body/DisposableTip/Visuals/Tip"
TIP_MATERIAL = "Looks/P6TipColor"

# ---- cameras --------------------------------------------------------------------------------------------------------
SIDE_CAM_EYE = (1.15, 1.45, 1.35)
SIDE_CAM_TARGET = (-0.05, 0.55, 1.05)
WRIST_CAM_EYE_LOCAL = {"R": (0.0, -0.13, 0.02), "L": (0.0, 0.13, 0.02)}
WRIST_CAM_TARGET_LOCAL = {"R": (-0.1248, 0.0846, -0.0096), "L": (0.0, -0.30, -0.06)}
WRIST_CAM_FOCAL = {"R": 12.0, "L": 8.0}
THUMB_CAM_EYE_LOCAL = (0.07, 0.0, 0.09)         # right wrist frame, m
THUMB_CAM_TARGET_LOCAL = (0.06, 0.035, -0.065)
THUMB_CAM_UP_LOCAL = (1.0, 0.0, 0.0)            # image up = wrist +x
THUMB_CAM_FOCAL = 12.0

CAMERAS = {"ego_view": "head_camera", "side_view": "side_camera", "left_wrist_view": "left_wrist_camera",
           "right_wrist_view": "right_wrist_camera", "right_thumb_view": "right_thumb_camera"}
"""Camera view name -> scene attribute."""
STATE_GROUPS = {"left_arm": {"start": 0, "end": 7}, "left_hand": {"start": 7, "end": 27},
                "right_arm": {"start": 27, "end": 34}, "right_hand": {"start": 34, "end": 54}}
"""Joint groups of the 54-D state and action: name -> {start, end} columns."""


def rgb8(frame: np.ndarray | torch.Tensor) -> np.ndarray:
    """A camera frame as contiguous uint8 RGB (H, W, 3)."""
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        scale = 255.0 if frame.size and float(frame.max()) <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def hole_local_xy(i: int, j: int) -> tuple[float, float]:
    """Pocket centre in the rack's local frame for staggered grid index (i, j)."""
    return (RACK_HOLE_BASE[0] + i * RACK_HOLE_PITCH[0],
            RACK_HOLE_BASE[1] + (j + (abs(i) % 2) * 0.5) * RACK_HOLE_PITCH[1])


def _add_lower_grip_collider(prim: Usd.Prim, prim_path: str) -> None:
    """The pipette's lower grip collider (PIP_BODY_SEAM_RING extruded over LOWER_GRIP_Z), invisible, with the asset's
    grip material; the body keeps the asset's mass and centre of mass."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    z0, z1 = LOWER_GRIP_Z
    col_path = f"{prim_path}/Body/Colliders/BodyLowerGripCollider"
    stage = prim.GetStage()
    ring = list(PIP_BODY_SEAM_RING)
    n = len(ring)
    mesh = UsdGeom.Mesh.Define(stage, col_path)
    pts = ([Gf.Vec3f(float(x), float(y), float(z0)) for x, y in ring]
           + [Gf.Vec3f(float(x), float(y), float(z1)) for x, y in ring])
    counts, idx = [], []
    for i in range(n):              # side quads with outward normals (the ring is counter-clockwise)
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    counts.append(n)
    idx += list(range(n - 1, -1, -1))   # bottom cap
    counts.append(n)
    idx += list(range(n, 2 * n))        # top cap
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([
        Gf.Vec3f(min(p[0] for p in pts), min(p[1] for p in pts), float(min(z0, z1))),
        Gf.Vec3f(max(p[0] for p in pts), max(p[1] for p in pts), float(max(z0, z1)))])
    mesh_prim = mesh.GetPrim()
    UsdGeom.Imageable(mesh_prim).MakeInvisible()
    schemas.define_collision_properties(col_path, sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=0.002, rest_offset=0.0))
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("convexHull")
    mat_path = f"{col_path}/material"
    mat_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=PIP_GRIP_MU_S, dynamic_friction=PIP_GRIP_MU_D,
                                             restitution=PIP_GRIP_RESTITUTION)
    mat_cfg.func(mat_path, mat_cfg)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)),
                                                      UsdShade.Tokens.strongerThanDescendants, "physics")
    mass_api = UsdPhysics.MassAPI.Apply(prim.GetChild("Body"))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, PIP_ASSET_COM_Z))
    mass_api.CreateMassAttr(float(PIP_ASSET_MASS))


def _color_tip(pipette_prim: Usd.Prim) -> None:
    """Bind an opaque TIP_COLOR surface (roughness 0.5) on the disposable-tip mesh (render only)."""
    from pxr import Gf, Sdf, UsdShade
    stage = pipette_prim.GetStage()
    root = pipette_prim.GetPath()
    tip = stage.GetPrimAtPath(root.AppendPath(TIP_MESH))
    if not tip.IsValid() or tip.GetTypeName() != "Mesh":
        raise RuntimeError(f"no tip mesh at {root}/{TIP_MESH}")
    mat = UsdShade.Material.Define(stage, root.AppendPath(TIP_MATERIAL))
    shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) for c in TIP_COLOR)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(tip).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    bound = UsdShade.MaterialBindingAPI(tip).ComputeBoundMaterial()[0].GetPath()
    if bound != mat.GetPath():
        raise RuntimeError(f"tip colour: {tip.GetPath()} resolves to {bound}, not {mat.GetPath()}")


def _wrap_spawn(spawn_cfg: SpawnerCfg, after: Callable[[Usd.Prim, str], None]) -> None:
    """Run `after(prim, prim_path)` right after the spawner of `spawn_cfg` (before Isaac Lab clones env 0)."""
    base = spawn_cfg.func

    def spawn(prim_path: str, cfg: SpawnerCfg, translation: tuple | None = None,
              orientation: tuple | None = None) -> Usd.Prim:
        prim = base(prim_path, cfg, translation, orientation)
        after(prim, prim_path)
        return prim

    spawn_cfg.func = spawn


def _heavy_dynamic(prim_path: str) -> None:
    """A heavy rigid body: finger brushes move it a little instead of producing solver impulses."""
    schemas.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True, disable_gravity=False, linear_damping=0.5, angular_damping=1.0,
        max_depenetration_velocity=1.0))
    schemas.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=STAND_MASS))


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, ...]:
    q = quat_from_matrix(create_rotation_matrix_from_view(torch.tensor([eye], dtype=torch.float32),
                                                          torch.tensor([target], dtype=torch.float32),
                                                          up_axis="Z", device="cpu"))[0]
    return tuple(float(v) for v in q)


def build_environment_cfg(layout: dict, device: str, side_camera: bool = True) -> TaskEnvCfg:
    """The env config for one layout {stand_dxy, pip_rail, rack_dxy, hole}; without the side camera if not
    `side_camera` (the policy does not see it)."""
    cfg = TaskEnvCfg()
    cfg.sim.device = device

    # stand: to the robot's right and forward, turned RACK_YAW_DEG; the pipette hangs at its calibrated spot, moved
    # along the rail by the layout
    stand_pos = (np.asarray(cfg.scene.pipette_stand.init_state.pos) + np.array([0.28, RACK_FORWARD_SHIFT, 0.0])
                 + np.array([layout["stand_dxy"][0], layout["stand_dxy"][1], 0.0]))
    stand_yaw = Rotation.from_euler("z", RACK_YAW_DEG, degrees=True)
    old_stand_rot = Rotation.from_quat(np.asarray(cfg.scene.pipette_stand.init_state.rot))
    old_pip_rot = Rotation.from_quat(np.asarray(cfg.scene.pipette_v5.init_state.rot))
    yaw_fix = stand_yaw * old_stand_rot.inv()
    pip_offset = yaw_fix.apply(np.asarray(cfg.scene.pipette_v5.init_state.pos)
                               - np.asarray(cfg.scene.pipette_stand.init_state.pos))
    cfg.scene.pipette_stand.init_state.pos = tuple(stand_pos)
    cfg.scene.pipette_stand.init_state.rot = tuple(stand_yaw.as_quat())
    cfg.scene.pipette_v5.init_state.pos = tuple(stand_pos + pip_offset + layout["pip_rail"] * _RAIL_DIR_W)
    cfg.scene.pipette_v5.init_state.rot = tuple((yaw_fix * old_pip_rot).as_quat())

    # pipette: lower grip collider, then the tip colour
    _wrap_spawn(cfg.scene.pipette_v5.spawn, _add_lower_grip_collider)
    _wrap_spawn(cfg.scene.pipette_v5.spawn, lambda prim, _path: _color_tip(prim))

    stand_spawn = cfg.scene.pipette_stand.spawn
    _wrap_spawn(stand_spawn, lambda _prim, path: _heavy_dynamic(path))
    cfg.scene.pipette_stand = RigidObjectCfg(
        prim_path=cfg.scene.pipette_stand.prim_path, spawn=stand_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(stand_pos), rot=tuple(stand_yaw.as_quat())))

    # rack: the holder mesh with box colliders that leave the chosen pocket open
    cfg.scene.centrifuge_tube.spawn.obj_paths = (str(PROPS_DIR / "Centrifuge_Tube_holder_thruholes.obj"),)
    cfg.scene.centrifuge_tube.spawn.color_overrides = {"Centrifuge_Tube_holder": (0.25, 0.35, 0.55)}
    cfg.scene.centrifuge_tube.spawn.collision = "off"
    rack_spawn = cfg.scene.centrifuge_tube.spawn
    rack_xy = (TUBE_RACK_POS[0] + layout["rack_dxy"][0], TUBE_RACK_POS[1] + layout["rack_dxy"][1])
    hole_xy = hole_local_xy(*layout["hole"])
    tube_start = np.array([rack_xy[0] + hole_xy[0] + TUBE_SPAWN_SHIFT[0],
                           rack_xy[1] + hole_xy[1] + TUBE_SPAWN_SHIFT[1], TUBE_SPAWN_Z])

    def rack_colliders(_prim: Usd.Prim, prim_path: str) -> None:
        hx, hy = hole_xy
        wo = RACK_OPEN_HALF
        px_, py_ = RACK_PLATE_HALF
        box = sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01), visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002,
                                                             rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        n_box = 0
        for z0, z1 in RACK_COLLIDER_BANDS:
            zc, sz = 0.5 * (z0 + z1), z1 - z0
            for cx_, cy_, sx_, sy_ in ((0.5 * (-px_ + hx - wo), 0.0, (hx - wo) + px_, 2 * py_),
                                       (0.5 * (hx + wo + px_), 0.0, px_ - (hx + wo), 2 * py_),
                                       (hx, 0.5 * (-py_ + hy - wo), 2 * wo, (hy - wo) + py_),
                                       (hx, 0.5 * (hy + wo + py_), 2 * wo, py_ - (hy + wo))):
                b = box.replace(size=(sx_, sy_, sz))
                b.func(f"{prim_path}/rack_collider_{n_box}", b, translation=(cx_, cy_, zc))
                n_box += 1
        _heavy_dynamic(prim_path)

    _wrap_spawn(rack_spawn, rack_colliders)
    cfg.scene.centrifuge_tube = RigidObjectCfg(
        prim_path=cfg.scene.centrifuge_tube.prim_path, spawn=rack_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(rack_xy[0], rack_xy[1], cfg.scene.centrifuge_tube.init_state.pos[2]),
                                                  rot=(0.0, 0.0, 0.0, 1.0)))

    # tube: standing in its pocket from step 0; friction combine "min" on the tube and the table top
    tube_spawn = sim_utils.CylinderCfg(
        radius=TUBE_RADIUS, height=TUBE_HEIGHT, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=TUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.15), metallic=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min", restitution_combine_mode="min",
            static_friction=LEFT_HAND_MU, dynamic_friction=LEFT_HAND_MU, restitution=0.0))
    table = cfg.scene.tabletop_collider.spawn
    table.physics_material = table.physics_material.replace(friction_combine_mode="min")

    def tube_com(prim: Usd.Prim, _path: str) -> None:
        from pxr import Gf, UsdPhysics
        UsdPhysics.MassAPI.Apply(prim).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, TUBE_COM_LOCAL_Z))

    _wrap_spawn(tube_spawn, tube_com)
    cfg.scene.tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube", spawn=tube_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(tube_start), rot=(0.0, 0.0, 0.0, 1.0)))

    # cameras: side (optional), both wrists, right thumb; the ego (head) camera is the task's
    if side_camera:
        cfg.scene.side_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera", update_period=0.0, height=360, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=SIDE_CAM_EYE, rot=_look_at(SIDE_CAM_EYE, SIDE_CAM_TARGET), convention="opengl"))
    for arm, attr in (("L", "left_wrist_camera"), ("R", "right_wrist_camera")):
        setattr(cfg.scene, attr, CameraCfg(
            prim_path=f"{WRIST_PRIM_PATH[arm]}/WristCamera", update_period=0.0, height=360, width=480,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=WRIST_CAM_FOCAL[arm], clipping_range=(0.03, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_EYE_LOCAL[arm],
                                       rot=_look_at(WRIST_CAM_EYE_LOCAL[arm], WRIST_CAM_TARGET_LOCAL[arm]),
                                       convention="opengl")))
    e, t = np.asarray(THUMB_CAM_EYE_LOCAL, float), np.asarray(THUMB_CAM_TARGET_LOCAL, float)
    z_axis = -(t - e) / np.linalg.norm(t - e)
    x_axis = np.cross(np.asarray(THUMB_CAM_UP_LOCAL, float), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    q_thumb = quat_from_matrix(torch.tensor(np.stack([x_axis, y_axis, z_axis], axis=1)[None], dtype=torch.float32))[0]
    cfg.scene.right_thumb_camera = CameraCfg(
        prim_path=f"{WRIST_PRIM_PATH['R']}/ThumbCamera", update_period=0.0, height=360, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=THUMB_CAM_FOCAL, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=THUMB_CAM_EYE_LOCAL, rot=tuple(float(v) for v in q_thumb), convention="opengl"))

    # finger-pipette contact sensors (part of the recorded scene graph)
    for name, chain in RIGHT_FINGER_DISTAL:
        setattr(cfg.scene, f"{name}_pipette_contact", ContactSensorCfg(
            prim_path=f"{WRIST_PRIM_PATH['R']}/{chain}/{name}", update_period=0.0, history_length=10,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/PipetteV5/Body"]))
    cfg.scene.r_wrist_pipette_contact = ContactSensorCfg(
        prim_path=WRIST_PRIM_PATH["R"], update_period=0.0, history_length=10,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/PipetteV5/Body"])
    return cfg
