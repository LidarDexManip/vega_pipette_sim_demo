"""Re-render a logged run in high definition: every camera at --cam-scale times its resolution.

The scene of the run's layout (from run.json) is built and settled as for the live run. Then each logged row is written into it (robot
joints and the props' poses: pipette, plunger, tube, rack and stand for the pipette task; tube and rack for the tube
pick-and-place), the kinematics are updated and every camera renders; no physics step runs, so the replay shows
exactly the logged motion.

  python -m vega_pipette.video.render_hd --run RUN_DIR [--cam-scale 3] [--device cuda:0]

RUN_DIR is a live run of either task (objlog.npz, run.json). Writes RUN_DIR/hd/<view>.mp4 for every camera of the
task's scene, one frame per row up to the row the evaluation would stop on (progress/rollout.py,
progress/pick_place.py).
The tube pick-and-place scene is built with CPU physics, as in its live runs (--device is the pipette task's).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..progress import pick_place, rollout
from ..sim import look
from ..sim.layouts import parse_layout, parse_pick_place_layout
from .recorder import Recorder

PLAYBACK_KEYS = ("state", "pip_pos", "pip_quat", "tube_pos", "tube_quat", "plunger", "rack_pos", "stand_pos")
PICK_PLACE_KEYS = ("state", "tube_pos", "tube_quat", "rack_pos", "rack_quat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a run directory (objlog.npz, run.json)")
    ap.add_argument("--cam-scale", type=int, default=3, help="camera resolution factor (default 3)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: RUN/hd")
    ap.add_argument("--device", default="cuda:0",
                    help="simulation device of a pipette-task replay (default cuda:0; the replay steps no physics)")
    args = ap.parse_args()
    run = args.run.resolve()
    with np.load(run / "objlog.npz") as z:
        tube_task = "hole_a" in z.files
    device = "cpu" if tube_task else args.device
    from isaaclab.app import AppLauncher
    AppLauncher({"enable_cameras": True, "device": device, "kit_args": look.KIT_ARG})

    import torch
    from scipy.spatial.transform import Rotation

    from ..env import scene as sc
    from ..env.constants import ACTION_JOINT_NAMES
    from ..sim.session import ACTION_HZ

    out = (args.out_dir or run / "hd").resolve()
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(run / "objlog.npz")
    layout = json.loads((run / "run.json").read_text())["layout"]
    if tube_task:
        from ..env.pick_place import CAMERAS
        from ..sim.pick_place import PickPlaceSession
        n = pick_place.run_state(str(run / "objlog.npz"))["n"]
        log = {k: np.asarray(z[k], np.float64) for k in PICK_PLACE_KEYS}
        sim = PickPlaceSession(parse_pick_place_layout(layout), device, cam_scale=args.cam_scale, side_camera=True)
    else:
        from ..sim.session import Session
        CAMERAS = sc.CAMERAS
        n = rollout.run_state(str(run / "objlog.npz"))["n"]
        log = {k: np.asarray(z[k], np.float64) for k in PLAYBACK_KEYS}
        sim = Session(parse_layout(layout), device, cam_scale=args.cam_scale)
    sim.settle()
    base = sim.control.base
    joints = [sim.robot.find_joints(nm)[0][0] for nm in ACTION_JOINT_NAMES]
    dev = torch.device(device)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)[None]

    def vel(i: int, j: int, pos: str, quat: str | None = None) -> torch.Tensor:
        lin = (log[pos][j] - log[pos][i]) * ACTION_HZ
        ang = (Rotation.from_quat(log[quat][j]) * Rotation.from_quat(log[quat][i]).inv()).as_rotvec() * ACTION_HZ \
            if quat else np.zeros(3)
        return t(np.concatenate([lin, ang]))

    if tube_task:
        def write_row(i: int) -> None:
            """Row i's logged state into the scene; velocities are the finite difference to the next row."""
            j = min(i + 1, n - 1)
            sim.robot.write_joint_state_to_sim_index(position=t(log["state"][i]),
                                                     velocity=t((log["state"][j] - log["state"][i]) * ACTION_HZ),
                                                     joint_ids=joints)
            for asset, key in ((sim.tube, "tube"), (sim.rack, "rack")):
                asset.write_root_pose_to_sim_index(root_pose=t(np.r_[log[f"{key}_pos"][i], log[f"{key}_quat"][i]]))
                asset.write_root_velocity_to_sim_index(root_velocity=vel(i, j, f"{key}_pos", f"{key}_quat"))
    else:
        plunger = sim.pipette.find_joints("PlungerJoint")[0][0]
        rack_q = t(sim.rack.data.root_quat_w[0].detach().cpu().numpy())[0]       # the rack and stand keep their settled
        stand_q = t(sim.stand.data.root_quat_w[0].detach().cpu().numpy())[0]     # orientation; only positions are logged

        def write_row(i: int) -> None:
            """Row i's logged state into the scene; velocities are the finite difference to the next row."""
            j = min(i + 1, n - 1)
            sim.robot.write_joint_state_to_sim_index(position=t(log["state"][i]),
                                                     velocity=t((log["state"][j] - log["state"][i]) * ACTION_HZ),
                                                     joint_ids=joints)
            sim.pipette.write_root_pose_to_sim_index(root_pose=t(np.r_[log["pip_pos"][i], log["pip_quat"][i]]))
            sim.pipette.write_root_velocity_to_sim_index(root_velocity=vel(i, j, "pip_pos", "pip_quat"))
            dp = (log["plunger"][j] - log["plunger"][i]) * ACTION_HZ
            sim.pipette.write_joint_state_to_sim_index(position=t([log["plunger"][i]]), velocity=t([dp]),
                                                       joint_ids=[plunger])
            sim.tube.write_root_pose_to_sim_index(root_pose=t(np.r_[log["tube_pos"][i], log["tube_quat"][i]]))
            sim.tube.write_root_velocity_to_sim_index(root_velocity=vel(i, j, "tube_pos", "tube_quat"))
            sim.rack.write_root_pose_to_sim_index(root_pose=torch.cat([t(log["rack_pos"][i])[0], rack_q])[None])
            sim.rack.write_root_velocity_to_sim_index(root_velocity=vel(i, j, "rack_pos"))
            sim.stand.write_root_pose_to_sim_index(root_pose=torch.cat([t(log["stand_pos"][i])[0], stand_q])[None])
            sim.stand.write_root_velocity_to_sim_index(root_velocity=vel(i, j, "stand_pos"))

    rec = Recorder(out, ACTION_HZ)
    for i in range(n):
        write_row(i)
        base.sim.forward()
        base.sim.render()
        base.scene.update(dt=base.step_dt)
        rec.add({view: sc.rgb8(base.scene[name].data.output["rgb"][0].detach().cpu().numpy())
                 for view, name in CAMERAS.items()})
        if i % 150 == 0:
            print(f"[hd   ] row {i} / {n}", flush=True)
    rec.close()
    print(f"[hd   ] {n} rows, {len(CAMERAS)} cameras at {args.cam_scale}x -> {out}", flush=True)
    sim.close()
    os._exit(0)                     # closing the app can take minutes; nothing is left to do


if __name__ == "__main__":
    try:
        main()
    except Exception:                # Isaac Sim's shutdown would turn a crash into exit code 0
        import traceback
        traceback.print_exc()
        os._exit(1)
