"""The Vega-U + Wuji Hand 2 articulation: PhysX solver settings, drives and the start pose (quaternions xyzw)."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .constants import ROBOT_USD

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.02, angular_damping=0.02, retain_accelerations=False, max_depenetration_velocity=2.0),
        # GPU PhysX uses the scene-wide max solver iterations; 32/4 on the robot and the pipette makes CPU physics
        # match the GPU dynamics the policy was trained on
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.7071068, 0.7071068),        # yawed +90 deg: facing the table at +y
        joint_pos={
            "Lift": 0.15, "torso_flip": 0.0, "head_j1": 0.35, "head_j2": 0.0, "head_j3": 0.0,   # head_j1: look down
            "L_arm_j1": 0.20, "L_arm_j2": 0.35, "L_arm_j3": 0.0, "L_arm_j4": -1.05, "L_arm_j5": 0.0, "L_arm_j6": 0.72,
            "L_arm_j7": 0.0,
            "R_arm_j1": -0.20, "R_arm_j2": -0.35, "R_arm_j3": 0.0, "R_arm_j4": -1.05, "R_arm_j5": 0.0, "R_arm_j6": -0.72,
            "R_arm_j7": 0.0,
            "[lr]_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "lift_torso": ImplicitActuatorCfg(joint_names_expr=["Lift", "torso_flip"], stiffness=300.0, damping=20.0),
        "head": ImplicitActuatorCfg(joint_names_expr=["head_j[1-3]"], stiffness=120.0, damping=10.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=["[LR]_arm_j[1-7]"], stiffness=1000.0, damping=50.0),
        "hands": ImplicitActuatorCfg(joint_names_expr=["[lr]_.*"], stiffness=2900.0, damping=114.592),
    },
)
"""The robot at the origin. The lift, torso and head are additionally pinned by joint limits during a run
(sim/session.py)."""
