"""Run the policy live in the Vega-U + Wuji pipette scene.

The robot takes the pipette off its stand, picks the centrifuge tube from the rack and presses the pipette into it,
driven by a policy server over ZeroMQ (control/policy_client.py). Actions run at 30 Hz; a new chunk of actions is
requested every inference period, synchronously or in the background (control/chunking.py). The task progress is
scored as the run goes (progress/rollout.py); the run stops a few seconds after its outcome (success or a failure),
when the window is closed, on Ctrl+C, or after --max-steps. It writes run.json and objlog.npz.

  python -m vega_pipette.live --gui --realtime                   # simulator window + live dashboard, wall-clock pace
  python -m vega_pipette.live --record-video                     # headless, one mp4 per camera
  python -m vega_pipette.live --gui --inference-mode async       # background inference
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from . import paths
from .progress.rollout import RolloutScore, frame_state, milestones
from .sim import look
from .sim.layouts import random_layout

TASK = "Pick up the centrifuge tube from the rack and press the pipette into it."
"""The task sentence the released policy was trained with."""
LANG_KEY = "annotation.human.action.task_description"
ACTION_HZ = 30.0
STOP_AFTER_STEPS = 90
"""Steps run after the outcome before the run stops (3 s of task time)."""
DASHBOARD_HZ = 10.0
"""How often the dashboard's state is recomputed (wall clock)."""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="zmq://127.0.0.1:5555", help="policy server, zmq://host:port")
    ap.add_argument("--task", default=TASK, help="task sentence sent with every query")
    ap.add_argument("--gui", action="store_true", help="show the simulator window")
    ap.add_argument("--realtime", action="store_true", help="pace actions to the wall clock and print the real-time factor")
    ap.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=None,
                    help="serve the live dashboard (default: on with --gui, off without)")
    ap.add_argument("--dashboard-port", type=int, default=8765, help="the dashboard's port (default 8765)")
    ap.add_argument("--record-video", action="store_true", help="write one mp4 per camera, side view included")
    ap.add_argument("--device", default="cpu", help="physics device: cpu (default) or cuda:0")
    ap.add_argument("--inference-mode", choices=("sync", "async"), default="sync",
                    help="sync: wait for each chunk; async: query in the background while the robot moves")
    ap.add_argument("--inference-hz", type=float, default=ACTION_HZ / 16,
                    help="chunk requests per second of task time (default 1.875: a new chunk every 16 actions)")
    ap.add_argument("--latency-steps", type=int, default=None,
                    help="async: emulate this fixed policy latency, in action steps")
    ap.add_argument("--lead", type=int, default=None,
                    help="async: request this many steps early (default: measured latency in steps + 2)")
    ap.add_argument("--max-steps", type=int, default=1200, help="actions to run at most (default 1200 = 40 s)")
    ap.add_argument("--keep-running", action="store_true", help="do not stop after the outcome")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: $VEGA_RUNS/live/<time>")
    a = ap.parse_args()
    if a.latency_steps is not None and a.inference_mode != "async":
        ap.error("--latency-steps needs --inference-mode async")
    if a.dashboard is None:
        a.dashboard = a.gui
    return a


class Pacer:
    """Holds each action step to 1/hz of wall time. A late step moves the schedule; it never catches up by shortening
    later steps."""

    def __init__(self, hz: float) -> None:
        self.dt = 1.0 / hz
        self.next_t = time.perf_counter()

    def tick(self) -> None:
        now = time.perf_counter()
        self.next_t += self.dt
        if now < self.next_t:
            time.sleep(self.next_t - now)
        else:
            self.next_t = now


class RateMeter:
    """The real-time factor over the last second: task time advanced / wall time."""

    def __init__(self) -> None:
        self.win_t, self.win_steps, self.value = time.perf_counter(), 0, None

    def tick(self) -> bool:
        """Count one step; True when a new value was measured (about once a second)."""
        self.win_steps += 1
        now = time.perf_counter()
        if now - self.win_t < 1.0:
            return False
        self.value = self.win_steps / ACTION_HZ / (now - self.win_t)
        self.win_t, self.win_steps = now, 0
        return True


def outcome_text(S: dict) -> str:
    """The outcome and the time each milestone was reached."""
    reached = ", ".join(f"{m['code']} {m['row'] / ACTION_HZ:.1f} s" for m in milestones(S, S["n"] - 1) if m["row"] is not None)
    return f"{'success' if S['success'] else S['stop']} ({reached or 'no milestone'})"


def fmt_layout(lay: dict) -> str:
    """One line: the offsets in cm and the tube's pocket."""
    cm = lambda v: ", ".join(f"{100 * x:+.1f}" for x in v)  # noqa: E731
    return (f"stand ({cm(lay['stand_dxy'])}) cm, pipette {100 * lay['pip_rail']:+.1f} cm on the rail, "
            f"rack ({cm(lay['rack_dxy'])}) cm, pocket {tuple(lay['hole'])}")


def main() -> None:
    args = parse_args()
    layout = random_layout()
    physics = "CPU" if args.device == "cpu" else "GPU"
    dash = None
    if args.dashboard:
        from .dashboard.server import Dashboard
        from .dashboard.state import snapshot
        dash = Dashboard(args.dashboard_port)
        print(f"[live ] dashboard at {dash.url}", flush=True)

    from isaaclab.app import AppLauncher
    launch = {"enable_cameras": True, "device": args.device, "kit_args": look.KIT_ARG}
    if args.gui:
        launch["visualizer"] = ["kit"]
    app = AppLauncher(launch).app

    from .control.chunking import ChunkScheduler
    from .control.policy_client import PolicyClient
    from .sim.objlog import ObjLog
    from .sim.session import STATE_GROUPS, Session

    exec_steps = int(round(ACTION_HZ / args.inference_hz))
    out = args.out_dir or paths.RUNS / "live" / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print(f"[live ] random layout {fmt_layout(layout)}, {physics} physics, inference every "
          f"{exec_steps} actions ({args.inference_mode}) -> {out}", flush=True)

    sim = Session(layout, args.device, window=args.gui, side_camera=args.dashboard or args.record_video)
    sim.settle()
    poses0 = sim.poses()

    client = PolicyClient.from_url(args.policy)
    contract = client.get_modality_config()
    views = [str(k).split(".", 1)[-1] for k in contract["video_keys"]]
    missing = [v for v in views if v not in sim.views]
    if missing:
        raise SystemExit(f"the policy asks for camera views {missing} that the scene does not render")
    client.reset()

    def query(obs: tuple[dict, np.ndarray, int]) -> np.ndarray:
        frames, state, step = obs
        reply = client.get_action({
            "video": {v: np.ascontiguousarray(frames[v])[None, None] for v in views},
            "state": {g: state[s["start"]:s["end"]].astype(np.float32)[None, None] for g, s in STATE_GROUPS.items()},
            "language": {LANG_KEY: [args.task]}}, options={"step": int(step)})
        return client.to_chunk(reply, STATE_GROUPS)

    sched = ChunkScheduler(query, exec_steps, args.inference_mode, args.latency_steps, args.lead,
                           log=lambda m: print(m, flush=True))
    log = ObjLog(poses0["pip_pos"], poses0["tube_pos"])
    score = RolloutScore(poses0["pip_pos"], poses0["tube_pos"])
    pacer = Pacer(ACTION_HZ) if args.realtime else None
    rate = RateMeter()
    recorder = None
    if args.record_video:
        from .video.recorder import Recorder
        recorder = Recorder(out, ACTION_HZ)
    stop = {"reason": "max_steps"}
    signal.signal(signal.SIGINT, lambda *_: stop.update(reason="interrupted"))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="terminated"))
    applied: list[int] = []

    def dashboard_state(ended: bool = False) -> dict:
        S = frame_state(log.arrays(), score.result())
        d = snapshot(S, outcome=score.stopped or ended)
        d.update(title="π0.5", subtitle=f"{args.inference_mode} · {physics} physics",
                 t=round(len(log) / ACTION_HZ, 1), rtf=None if rate.value is None else round(rate.value, 2),
                 infer_ms=round(sched.records[-1].latency_ms) if sched.records else None,
                 infer_steps=int(np.ceil(sched.records[-1].latency_ms * ACTION_HZ / 1000.0)) if sched.records else None,
                 chunk_every=applied[-1] - applied[-2] if len(applied) > 1 else None, ended=ended)
        return d

    print(f"[live ] running: {args.task!r}", flush=True)
    t0 = last_push = time.perf_counter()
    outcome_step = None
    for step in range(args.max_steps):
        state, poses = sim.state(), sim.poses()
        n_rec = len(sched.records)
        action = sched.act(step, lambda: (dict(sim.frames), state, step))
        new = [r for r in sched.records[n_rec:] if r.apply_step == step]
        if new:
            applied.append(step)
        log.add(state, action, poses, sum(r.latency_ms for r in new))
        score.add(log.row(step))
        if recorder is not None:
            recorder.add(sim.frames)
        if dash is not None:
            dash.push_frames(sim.frames)
            if time.perf_counter() - last_push >= 1.0 / DASHBOARD_HZ:
                dash.push_state(dashboard_state())
                last_push = time.perf_counter()
        sim.step(action)
        if pacer is not None:
            pacer.tick()
        if rate.tick() and args.realtime:
            slow = "" if rate.value >= 0.95 else "  (slower than real time)"
            print(f"[rt   ] t {(step + 1) / ACTION_HZ:6.1f} s  real-time factor {rate.value:4.2f}{slow}", flush=True)
        if not args.realtime and step % 300 == 0:
            print(f"[live ] step {step:5d}  task time {step / ACTION_HZ:5.1f} s  wall {time.perf_counter() - t0:5.0f} s",
                  flush=True)
        if score.stopped and outcome_step is None:
            outcome_step = step
            print(f"[live ] outcome at {score.n / ACTION_HZ:.1f} s: {outcome_text(frame_state(log.arrays(), score.result()))}",
                  flush=True)
        if outcome_step is not None and not args.keep_running and step - outcome_step >= STOP_AFTER_STEPS:
            stop["reason"] = "outcome"
        if stop["reason"] != "max_steps":
            break
        if args.gui and not app.is_running():
            stop["reason"] = "window_closed"
            break
    sched.close()
    if recorder is not None:
        recorder.close()
    log.save(out / "objlog.npz", sched.records, sched.hold_steps)
    final = frame_state(log.arrays(), score.result())
    if dash is not None:
        dash.push_state(dashboard_state(ended=True))
    (out / "run.json").write_text(json.dumps({
        "layout": layout, "task": args.task, "device": args.device, "inference_mode": args.inference_mode,
        "action_hz": ACTION_HZ, "inference_hz": ACTION_HZ / exec_steps, "latency_steps": args.latency_steps,
        "lead": sched.lead, "steps": len(log), "stop": stop["reason"],
        "outcome": outcome_text(final), "wall_s": round(time.perf_counter() - t0, 1), "queries": len(sched.records),
        "held_steps": len(sched.hold_steps), "server": contract.get("server")}, indent=1, default=str))
    print(f"[live ] done: {len(log)} actions ({len(log) / ACTION_HZ:.1f} s of task time), stop {stop['reason']}, "
          f"{outcome_text(final)} -> {out}", flush=True)
    if dash is not None:
        time.sleep(1.0)             # let the page receive the final state
    sim.close()
    os._exit(0)                     # closing the Isaac app can hang for minutes; nothing is left to do


if __name__ == "__main__":
    main()
