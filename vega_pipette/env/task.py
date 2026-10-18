"""The lab scene and the manager-based environment config of the task (PhysX, 30 Hz control, one environment).

The robot stands at the origin facing a packing table (work surface at SURFACE_Z) with a pipette stand and pipette,
a tube rack, a mini centrifuge and an open pot on it, inside a warehouse room. env/scene.py adds what depends on the
layout (stand, pipette, rack, tube) and the policy cameras. Quaternions are xyzw.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from . import events, observations
from .constants import (
    ACTION_JOINT_NAMES,
    EGO_CAMERA_HEIGHT,
    EGO_CAMERA_WIDTH,
    LAB_PROPS_DIR,
    PACKING_TABLE_USD,
    PIPETTE_USD,
    SURFACE_Z,
)
from .objects.external_mesh import ExternalMeshObjectCfg
from .objects.pipette_stand import PipetteStandObjectCfg
from .objects.pot import PotObjectCfg
from .robot import ROBOT_CFG

_IDENTITY = (0.0, 0.0, 0.0, 1.0)

STAND_XY = (-0.05, 0.60)
STAND_ROT = (0.0, 0.0, -0.8660254, 0.5)                          # yaw -120 deg
PIPETTE_RACK_HEIGHT = 0.285
PIPETTE_OFFSET_FROM_STAND = (-0.0190440, -0.0333680, 0.2494800)  # the pipette hanging still on the rail
PIPETTE_ROT = (0.0067900, 0.0278890, -0.9653400, -0.2594140)
POT_POS = (-0.15 + 0.30, 0.40 - 0.10, SURFACE_Z)       # computed as recorded (0.40 - 0.10 != 0.30 in floats)
MINI_CENTRIFUGE_POS = (-0.32, 0.575, SURFACE_Z)
TUBE_RACK_POS = (-0.45, 0.35, SURFACE_Z)
RIGHT_GRIP_MU = 1.5            # right hand and pipette friction
LEFT_HAND_MU = 3.5             # left hand and tube friction (the tube must not slip)
TABLE_POSES = {"PackingTable": (0.0, 0.55, -0.2), "PackingTable_2": (-3.5, 0.55, -0.2),
               "PackingTable_3": (3.5, 0.55, -0.2), "PackingTable_4": (3.5, -5.0, -0.2),
               "PackingTable_5": (-3.5, -5.0, -0.2), "PackingTable_6": (0.0, -5.0, -0.2)}


def _packing_table(prim_name: str) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POSES[prim_name], rot=_IDENTITY),
        spawn=UsdFileCfg(usd_path=str(PACKING_TABLE_USD),
                         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )


@configclass
class LabSceneCfg(InteractiveSceneCfg):
    """The scene entities, in their creation order."""

    robot: ArticulationCfg = ROBOT_CFG

    room_walls = AssetBaseCfg(          # render only: the warehouse's colliders are switched off
        prim_path="{ENV_REGEX_NS}/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=_IDENTITY),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
                         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=800.0))

    packing_table = _packing_table("PackingTable")
    packing_table_2 = _packing_table("PackingTable_2")
    packing_table_3 = _packing_table("PackingTable_3")
    packing_table_4 = _packing_table("PackingTable_4")
    packing_table_5 = _packing_table("PackingTable_5")
    packing_table_6 = _packing_table("PackingTable_6")

    # the work table's collider: the packing-table USD's own collider sits on an instanceable prim
    tabletop_collider = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TabletopCollider",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, SURFACE_Z - 0.15), rot=_IDENTITY),
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.8, 0.30), visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="max", restitution_combine_mode="min",
                                                           static_friction=1.0, dynamic_friction=1.0, restitution=0.0)),
    )

    pot = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Pot", spawn=PotObjectCfg(),
                         init_state=RigidObjectCfg.InitialStateCfg(pos=POT_POS, rot=_IDENTITY))

    pipette_stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PipetteStand",
        spawn=PipetteStandObjectCfg(collision="convex_decomposition", height=PIPETTE_RACK_HEIGHT, width=0.270,
                                    rail_height=0.055, rail_length=0.250, rail_z_offset=-0.005),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(STAND_XY[0], STAND_XY[1], SURFACE_Z), rot=STAND_ROT),
    )

    pipette_v5: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/PipetteV5",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PIPETTE_USD), activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(      # 32/4, as the robot (see env/robot.py)
                enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(s + o for s, o in zip((STAND_XY[0], STAND_XY[1], SURFACE_Z), PIPETTE_OFFSET_FROM_STAND)),
            rot=PIPETTE_ROT, joint_pos={"PlungerJoint": 0.0}, joint_vel={"PlungerJoint": 0.0}),
        actuators={"plunger": ImplicitActuatorCfg(joint_names_expr=["PlungerJoint"], stiffness=120.0, damping=1.8,
                                                  effort_limit_sim=12.0, velocity_limit_sim=1.0)},
    )

    mini_centrifuge = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MiniCentrifuge",
        spawn=ExternalMeshObjectCfg(
            obj_paths=(str(LAB_PROPS_DIR / "mini_centrifuge__laboratory__lowpoly.glb"),), up_axis="Y",
            target_height=0.175, use_embedded_textures=True, collision="trimesh", metallic=0.0, roughness=0.5,
            default_color=(0.85, 0.85, 0.87)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=MINI_CENTRIFUGE_POS, rot=_IDENTITY),
    )

    centrifuge_tube = AssetBaseCfg(     # the tube rack; env/scene.py swaps in its mesh with open pockets
        prim_path="{ENV_REGEX_NS}/CentrifugeTube",
        spawn=ExternalMeshObjectCfg(
            obj_paths=tuple(str(LAB_PROPS_DIR / f"{part}.obj") for part in
                            ("Centrifuge_Tube", "Centrifuge_Tube_Lid", "Centrifuge_Tube_liquid", "Centrifuge_Tube_holder")),
            up_axis="Y", unit_scale=0.01, scale_multiplier=0.55, collision="trimesh", metallic=0.0, roughness=0.5,
            default_color=(0.85, 0.85, 0.88),
            color_overrides={"Centrifuge_Tube": (0.90, 0.92, 0.95), "Centrifuge_Tube_Lid": (0.20, 0.45, 0.85),
                             "Centrifuge_Tube_liquid": (0.25, 0.70, 0.35), "Centrifuge_Tube_holder": (0.25, 0.35, 0.55)}),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TUBE_RACK_POS, rot=_IDENTITY),
    )

    # the ego camera: the left lens of the head's stereo camera, deep in the robot's link tree
    head_camera = CameraCfg(
        prim_path=("{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_link/torso_flip_link/"
                   "arm_center/head_l1/head_l2/head_l3/zed_left_camera/TaskCamera"),
        update_period=0.0, height=EGO_CAMERA_HEIGHT, width=EGO_CAMERA_WIDTH, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=7.0, horizontal_aperture=20.0, clipping_range=(0.01, 4.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.360547, -0.330571, 0.608281, 0.625078), convention="world"),
    )


@configclass
class ActionsCfg:
    joint_targets = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ACTION_JOINT_NAMES, scale=1.0,
                                                    use_default_offset=False, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=observations.controlled_joint_pos,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINT_NAMES,
                                                                preserve_order=True)})
        joint_vel = ObsTerm(func=observations.controlled_joint_vel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINT_NAMES,
                                                                preserve_order=True)})
        rgb = ObsTerm(func=observations.head_rgb, params={"sensor_cfg": SceneEntityCfg("head_camera")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _friction(asset_cfg: SceneEntityCfg, mu: float) -> EventTerm:
    return EventTerm(func=base_mdp.randomize_rigid_body_material, mode="startup",
                     params={"asset_cfg": asset_cfg, "static_friction_range": (mu, mu), "dynamic_friction_range": (mu, mu),
                             "restitution_range": (0.0, 0.0), "num_buckets": 1})


_HAND = ("wrist", "thumb.*", "index_finger.*", "middle_finger.*", "ring_finger.*", "pinky.*")


@configclass
class EventCfg:
    reset_scene = EventTerm(func=events.reset_scene_to_default, mode="reset")
    pipette_friction = _friction(SceneEntityCfg("pipette_v5", body_names=".*"), RIGHT_GRIP_MU)
    right_hand_friction = _friction(SceneEntityCfg("robot", body_names=[f"r_{b}" for b in _HAND]), RIGHT_GRIP_MU)
    left_hand_friction = _friction(SceneEntityCfg("robot", body_names=[f"l_{b}" for b in _HAND]), LEFT_HAND_MU)


@configclass
class _NoTerms:
    """No reward and no termination terms: a run ends only when the runner stops it."""


@configclass
class TaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: LabSceneCfg = LabSceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=False)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: _NoTerms = _NoTerms()
    terminations: _NoTerms = _NoTerms()

    def __post_init__(self) -> None:
        self.decimation = 4                     # 30 Hz control, 120 Hz physics, one render per control step
        self.episode_length_s = 1.0e6
        self.sim.dt = 1.0 / 120.0
        self.sim.physics = PhysxCfg()
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
