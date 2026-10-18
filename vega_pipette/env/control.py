"""The environment behind a numpy interface: 54-D absolute joint targets in, joint state and the ego image out."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import ACTION_JOINT_NAMES

if TYPE_CHECKING:
    import gymnasium as gym


class Control:
    """Wraps a ManagerBasedRLEnv of env/task.py. Actions are clipped to the robot's soft joint limits."""

    def __init__(self, env: gym.Env) -> None:
        self.base = env
        robot = env.scene["robot"]
        ids, resolved = robot.find_joints(ACTION_JOINT_NAMES, preserve_order=True)
        if resolved != ACTION_JOINT_NAMES:
            raise RuntimeError(f"action joint order mismatch:\nexpected={ACTION_JOINT_NAMES}\nresolved={resolved}")
        self._ids = ids
        limits = robot.data.soft_joint_pos_limits[0, ids].detach().cpu().numpy()
        self._lower, self._upper = limits[:, 0], limits[:, 1]

    @staticmethod
    def _public(obs: dict) -> dict[str, np.ndarray]:
        policy = obs.get("policy", obs)
        return {k: policy[k].detach().cpu().numpy() for k in ("joint_pos", "joint_vel", "rgb")}

    def reset(self) -> dict[str, np.ndarray]:
        obs, _ = self.base.reset()
        return self._public(obs)

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Apply one action (54,) and advance one control step."""
        target = np.asarray(action, dtype=np.float32).reshape(1, 54)
        target = np.clip(target, self._lower, self._upper)
        obs, *_ = self.base.step(self.base.scene["robot"].data.joint_pos.new_tensor(target))
        return self._public(obs)

    def default_action(self) -> np.ndarray:
        """The robot's default joint positions in the action order."""
        return self.base.scene["robot"].data.default_joint_pos[0, self._ids].detach().cpu().numpy().copy()

    def close(self) -> None:
        self.base.close()
