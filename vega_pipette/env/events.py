"""Event terms of the task."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Isaac Lab's reset_scene_to_default, except that kinematic rigid objects get their pose only: PhysX does not
    take a velocity for a kinematic body (on the CPU it reports an error for each such write)."""
    for obj in env.scene.rigid_objects.values():
        pose = obj.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        if not getattr(getattr(obj.cfg.spawn, "rigid_props", None), "kinematic_enabled", False):
            obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch[env_ids].clone(),
                                                 env_ids=env_ids)
    for art in env.scene.articulations.values():
        pose = art.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        art.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        art.write_root_velocity_to_sim_index(root_velocity=art.data.default_root_vel.torch[env_ids].clone(),
                                             env_ids=env_ids)
        art.write_joint_position_to_sim_index(position=art.data.default_joint_pos.torch[env_ids].clone(),
                                              env_ids=env_ids)
        art.write_joint_velocity_to_sim_index(velocity=art.data.default_joint_vel.torch[env_ids].clone(),
                                              env_ids=env_ids)
