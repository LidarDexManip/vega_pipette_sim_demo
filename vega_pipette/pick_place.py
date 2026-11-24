"""Run the tube pick-and-place policy live in the Vega-U + Wuji scene.

The left hand takes the tube out of pocket A of the rack and puts it into pocket B, driven by the pick-and-place policy
server over ZeroMQ (control/policy_client.py). The instruction switches from picking to placing to releasing on the
simulation's truth (progress/pick_place.py). While placing, it ends in the hand target the runner computes from the
robot's joints and the hole numbers (control/place_hint.py). Actions run at 30 Hz; a new chunk of actions is requested
every inference period, synchronously or in the background (control/chunking.py), and at once after every new
instruction. The run stops at its outcome (one second after the release), when the window is closed, on Ctrl+C, or
after --max-steps. It writes run.json and objlog.npz. Physics runs on the CPU, the physics the policy was trained and
evaluated with.

  python -m vega_pipette.pick_place --gui --realtime                   # simulator window + live dashboard
  python -m vega_pipette.pick_place --record-video                     # headless, one mp4 per camera
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
from .live import ACTION_HZ, DASHBOARD_HZ, LANG_KEY, Pacer, RateMeter
from .progress import pick_place as pp
from .progress.rack import column_row
from .sim import look
from .sim.layouts import random_pick_place_layout

EXEC_STEPS = 8
"""Actions run from each chunk by default (the policy plans 32)."""
DEVICE = "cpu"
"""The physics device: CPU, the physics the policy was trained and evaluated with."""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="zmq://127.0.0.1:5555", help="policy server, zmq://host:port")
    ap.add_argument("--gui", action="store_true", help="show the simulator window")
    ap.add_argument("--realtime", action="store_true",
                    help="pace actions to the wall clock and print the real-time factor")
    ap.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=None,
                    help="serve the live dashboard (default: on with --gui, off without)")
    ap.add_argument("--dashboard-port", type=int, default=8765, help="the dashboard's port (default 8765)")
    ap.add_argument("--record-video", action="store_true", help="write one mp4 per camera, side view included")
    ap.add_argument("--inference-mode", choices=("sync", "async"), default="sync",
                    help="sync: wait for each chunk; async: query in the background while the robot moves")
    ap.add_argument("--inference-hz", type=float, default=ACTION_HZ / EXEC_STEPS,
                    help=f"chunk requests per second of task time (default {ACTION_HZ / EXEC_STEPS:g}: a new chunk "
                         f"every {EXEC_STEPS} actions)")
    ap.add_argument("--latency-steps", type=int, default=None,
                    help="async: emulate this fixed policy latency, in action steps")
    ap.add_argument("--lead", type=int, default=None,
                    help="async: request this many steps early (default: measured latency in steps + 2)")
    ap.add_argument("--max-steps", type=int, default=1900, help="actions to run at most (default 1900 = 63 s)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: $VEGA_RUNS/live/<time>")
    a = ap.parse_args()
    if a.latency_steps is not None and a.inference_mode != "async":
        ap.error("--latency-steps needs --inference-mode async")
    if a.dashboard is None:
        a.dashboard = a.gui
    return a


def outcome_text(S: dict) -> str:
    """The outcome and the time each milestone was reached."""
    reached = ", ".join(f"{m['code']} {m['row'] / ACTION_HZ:.1f} s" for m in pp.milestones(S, S["n"] - 1)
                        if m["row"] is not None)
    return f"{pp.OUTCOME_TEXT[S['outcome']]} ({reached or 'no milestone'})"


def fmt_layout(lay: dict) -> str:
    """One line: the rack offset in cm and the two pockets."""
    (ca, ra), (cb, rb) = column_row(lay["hole_a"]), column_row(lay["hole_b"])
    return (f"rack ({100 * lay['rack_dxy'][0]:+.1f}, {100 * lay['rack_dxy'][1]:+.1f}) cm, "
            f"from column {ca} row {ra} to column {cb} row {rb}")


def main() -> None:
    args = parse_args()
    layout = random_pick_place_layout()
    dash = None
    if args.dashboard:
        from .dashboard.server import PICK_PLACE_SHEET, Dashboard
        from .dashboard.state import pick_place_snapshot
        dash = Dashboard(args.dashboard_port, sheet=PICK_PLACE_SHEET)
        print(f"[live ] dashboard at {dash.url}", flush=True)

    from isaaclab.app import AppLauncher
    launch = {"enable_cameras": True, "device": DEVICE, "kit_args": look.KIT_ARG}
    if args.gui:
        launch["visualizer"] = ["kit"]
    app = AppLauncher(launch).app

    from .control import place_hint
    from .control.chunking import ChunkScheduler
    from .control.policy_client import PolicyClient
    from .sim.objlog import PICK_PLACE_COLS, PICK_PLACE_EXTRA, ObjLog
    from .sim.pick_place import POLICY_VIEWS, PickPlaceSession
    from .sim.session import STATE_GROUPS

    exec_steps = int(round(ACTION_HZ / args.inference_hz))
    out = args.out_dir or paths.RUNS / "live" / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print(f"[live ] random layout {fmt_layout(layout)}, CPU physics, inference every {exec_steps} actions "
          f"({args.inference_mode}) -> {out}", flush=True)

    sim = PickPlaceSession(layout, DEVICE, window=args.gui, side_camera=args.dashboard or args.record_video)
    sim.settle()

    client = PolicyClient.from_url(args.policy)
    contract = client.get_modality_config()
    views = [str(k).split(".", 1)[-1] for k in contract["video_keys"]]
    missing = [v for v in views if v not in POLICY_VIEWS]
    if missing:
        raise SystemExit(f"the policy asks for camera views {missing}; this task's policy cameras are "
                         f"{list(POLICY_VIEWS)}")
    client.reset()
    prog = pp.PickPlaceProgress(layout["hole_a"], layout["hole_b"])
    hint = place_hint.PlaceHint(layout["hole_a"], layout["hole_b"])
    prompts: dict[int, str] = {}                 # observation step -> the instruction as the policy reads it
    target: list[np.ndarray | None] = [None]     # the hand target of the latest place query

    def instruction(step: int, state: np.ndarray, poses: dict) -> str:
        """The instruction of a query at this step; the place sentence carries the hint's hand target."""
        text = hint.sentence(prog.sentence, state, poses) if prog.stage == "place" else prog.sentence
        target[0] = place_hint.pocket_xy(text)
        prompts[step] = place_hint.as_read(text, state)
        return text

    def query(obs: tuple[dict, np.ndarray, int, str]) -> np.ndarray:
        frames, state, step, prompt = obs
        reply = client.get_action({
            "video": {v: np.ascontiguousarray(frames[v])[None, None] for v in views},
            "state": {g: state[s["start"]:s["end"]].astype(np.float32)[None, None] for g, s in STATE_GROUPS.items()},
            "language": {LANG_KEY: [prompt]}}, options={"step": int(step)})
        return client.to_chunk(reply, STATE_GROUPS)

    sched = ChunkScheduler(query, exec_steps, args.inference_mode, args.latency_steps, args.lead,
                           log=lambda m: print(m, flush=True))
    log = ObjLog({"hole_a": np.array(layout["hole_a"]), "hole_b": np.array(layout["hole_b"])}, PICK_PLACE_COLS,
                 PICK_PLACE_EXTRA)
    pacer = Pacer(ACTION_HZ) if args.realtime else None
    rate = RateMeter()
    recorder = None
    if args.record_video:
        from .video.recorder import Recorder
        recorder = Recorder(out, ACTION_HZ, pix_fmt="yuv444p")
    stop = {"reason": "max_steps"}
    signal.signal(signal.SIGINT, lambda *_: stop.update(reason="interrupted"))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="terminated"))
    applied: list[int] = []

    def hint_rows() -> dict:
        return {"hint_move": log.rows["hint_move"], "hint_shift": log.rows["hint_shift"],
                "prompt_step": sorted(prompts), "prompt_text": [prompts[s] for s in sorted(prompts)]}

    def dashboard_state(ended: bool = False) -> dict:
        d = pick_place_snapshot(pp.frame_state(prog, hint_rows()), outcome=prog.stopped or ended)
        d.update(title="π0.5", subtitle=f"{args.inference_mode} · CPU physics",
                 t=round(len(log) / ACTION_HZ, 1), rtf=None if rate.value is None else round(rate.value, 2),
                 infer_ms=round(sched.records[-1].latency_ms) if sched.records else None,
                 infer_steps=int(np.ceil(sched.records[-1].latency_ms * ACTION_HZ / 1000.0)) if sched.records else None,
                 chunk_every=applied[-1] - applied[-2] if len(applied) > 1 else None, ended=ended)
        return d

    print(f"[live ] running: {prog.sentence!r}", flush=True)
    t0 = last_push = time.perf_counter()
    for step in range(args.max_steps):
        state, poses = sim.state(), sim.poses()
        hint.observe(prog.stage, state, poses)
        changed = prog.add({"state": state, **poses})
        if changed:
            if prog.stopped:
                stop["reason"] = "outcome"
                break
            sched.restart()
        frames = dict(sim.frames)
        n_rec = len(sched.records)
        action = sched.act(step, lambda: (frames, state, step, instruction(step, state, poses)))
        if changed:
            said = ("hand open and away; the tube must stay seated for 1 s" if prog.stage == "check"
                    else repr(prompts.get(step, prog.sentence)))
            print(f"[live ] t {step / ACTION_HZ:5.1f} s: {said}", flush=True)
        new = [r for r in sched.records[n_rec:] if r.apply_step == step]
        if new:
            applied.append(step)
        placing = prog.stage == "place" and target[0] is not None
        log.add(state, action, {**poses,
                                "hint_move": place_hint.hand_move(target[0], state, poses) if placing
                                else np.full(2, np.nan),
                                "hint_shift": 1e3 * float(np.linalg.norm(hint.shift))
                                if placing and hint.shift is not None else np.nan},
                sum(r.latency_ms for r in new))
        if recorder is not None:
            recorder.add(frames)
        if dash is not None:
            dash.push_frames(frames)
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
            print(f"[live ] step {step:5d}  task time {step / ACTION_HZ:5.1f} s  "
                  f"wall {time.perf_counter() - t0:5.0f} s", flush=True)
        if stop["reason"] != "max_steps":
            break
        if args.gui and not app.is_running():
            stop["reason"] = "window_closed"
            break
    sched.close()
    if recorder is not None:
        recorder.close()
    log.save(out / "objlog.npz", sched.records, sched.hold_steps, prompts)
    final = pp.frame_state(prog)
    if dash is not None:
        dash.push_state(dashboard_state(ended=True))
    (out / "run.json").write_text(json.dumps({
        "task": "pick-place", "layout": layout, "instructions": prog.sentences, "device": DEVICE,
        "inference_mode": args.inference_mode, "action_hz": ACTION_HZ, "inference_hz": ACTION_HZ / exec_steps,
        "latency_steps": args.latency_steps, "lead": sched.lead, "steps": len(log), "stop": stop["reason"],
        "outcome": outcome_text(final), "events": prog.events, "timeouts": prog.timeouts, "place_hint": hint.record(),
        "wall_s": round(time.perf_counter() - t0, 1), "queries": len(sched.records),
        "held_steps": len(sched.hold_steps), "server": contract.get("server")}, indent=1, default=str))
    print(f"[live ] done: {len(log)} actions ({len(log) / ACTION_HZ:.1f} s of task time), stop {stop['reason']}, "
          f"{outcome_text(final)} -> {out}", flush=True)
    if dash is not None:
        time.sleep(1.0)             # let the page receive the final state
    sim.close()
    os._exit(0)                     # closing the Isaac app can hang for minutes; nothing is left to do


if __name__ == "__main__":
    main()
