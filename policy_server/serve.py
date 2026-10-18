#!/usr/bin/env python3
"""serve.py -- the Vega-U + Wuji pipette pi0.5 policy behind the live runner's ZeroMQ protocol (inference only).

    python policy_server/serve.py [--checkpoint DIR] [--port 5555] [--host 127.0.0.1] [--ready-file FILE]

Run it with the policy environment's python (README.md, Installation; VEGA_POLICY_PY in config/vega.env).
The checkpoint dir holds params/ and assets/vega_wuji_pipette/norm_stats.json (downloaded from Hugging Face, README.md).
The server loads the model, runs one warm-up inference (the first JAX compile takes a few minutes; later starts load
the compiled program from the cache), and only then binds the socket, so a client's first `ping` means "ready".

Protocol (one REP socket; every message is msgpack; numpy arrays travel as
{"__ndarray_class__": true, "as_npy": <bytes of np.save(array)>}; a failed request is answered {"error": "<text>"}):

    ping                -> {"status": "ok"}
    get_modality_config -> {"video_keys": ["video.<camera>", ...], "state_keys", "state_key_dims", "action_keys",
                            "action_key_dims", "action_horizon": 32, "state_norm": "quantile", "server": {...}}
    reset               -> starts a rollout (the query count of the log restarts at 0).  Every query draws fresh
                           sampler noise, N(0, 1).
    get_action          -> data {"observation": {"video": {camera: (1, 1, H, W, 3) uint8},
                                                 "state": {"left_arm": (1, 1, 7), "left_hand": (1, 1, 20),
                                                           "right_arm": (1, 1, 7), "right_hand": (1, 1, 20)} float32,
                                                 "language": {key: [prompt]}},
                                 "options": {"step": int}}
                           -> {"action.left_arm": (1, 32, 7), "action.left_hand": (1, 32, 20), "action.right_arm":
                               (1, 32, 7), "action.right_hand": (1, 32, 20)} float32, absolute joint targets
    kill                -> {"status": "bye"}, then the server exits

The server shares the GPU with Isaac Sim, so JAX does not preallocate GPU memory unless the caller sets
XLA_PYTHON_CLIENT_PREALLOCATE.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pathlib
import signal
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
_RUNS = pathlib.Path(os.environ.get("VEGA_RUNS") or REPO / "runs")
# Before jax / openpi are imported: share the GPU with the simulator, keep compiled programs and the tokenizer cache.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_RUNS / "jax_cache"))
os.environ.setdefault("OPENPI_DATA_HOME", str(_RUNS / "openpi_cache"))

from typing import Any  # noqa: E402

import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import vega_config  # noqa: E402
import zmq  # noqa: E402

SERVER_NAME = "vega-wuji-pipette-pi05-server/1"
DEFAULT_CHECKPOINT = pathlib.Path(os.environ.get("VEGA_CHECKPOINT") or REPO / "checkpoints" / "vega-wuji-pipette-pi05")
# The runner's 54-D state / action groups: [lo, hi) columns of the joint vector.
GROUPS_54 = {"left_arm": (0, 7), "left_hand": (7, 27), "right_arm": (27, 34), "right_hand": (34, 54)}


# ---------------------------------------------------------------------------------------------------- wire format
def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Cannot encode type {type(obj)}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, strict_types=False)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


# ---------------------------------------------------------------------------------------------------- helpers
def camera_names(slots: dict) -> list[str]:
    """{model slot: "observation.images.<camera>"} -> camera names in slot order (the runner's video keys)."""
    return [str(key).split("observation.images.", 1)[1] for key in slots.values()]


def contract(names: list[str], horizon: int, info: dict) -> dict:
    return {
        "video_keys": [f"video.{n}" for n in names],
        "state_keys": [f"state.{g}" for g in GROUPS_54],
        "state_key_dims": {f"state.{g}": hi - lo for g, (lo, hi) in GROUPS_54.items()},
        "action_keys": [f"action.{g}" for g in GROUPS_54],
        "action_key_dims": {f"action.{g}": hi - lo for g, (lo, hi) in GROUPS_54.items()},
        "action_horizon": int(horizon),
        "state_norm": "quantile",
        "server": info,
    }


def parse_observation(obs: dict, slots: dict) -> dict:
    """Runner observation -> openpi policy input {"images": {slot: uint8 HWC}, "state": (54,) f32, "prompt": str}.
    A missing camera, a wrong state width or a missing prompt is an error: the policy never runs on a guessed input."""
    if not isinstance(obs, dict):
        raise ValueError(f"observation is {type(obs).__name__}, not a dict")
    video, state, lang = obs.get("video") or {}, obs.get("state") or {}, obs.get("language") or {}
    images = {}
    for slot, key in slots.items():
        cam = str(key).split("observation.images.", 1)[1]
        if cam not in video:
            raise KeyError(f"camera {cam!r} (slot {slot}) missing from the observation; got {sorted(video)}")
        img = np.asarray(video[cam])
        while img.ndim > 3:              # (1, 1, H, W, 3): batch, then time; the newest frame
            img = img[-1] if img.ndim == 4 else img[0]
        if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
            raise ValueError(f"camera {cam!r}: need uint8 (H, W, 3), got {img.dtype} {img.shape}")
        images[slot] = img
    parts = []
    for g, (lo, hi) in GROUPS_54.items():
        if g not in state:
            raise KeyError(f"state group {g!r} missing; got {sorted(state)}")
        v = np.asarray(state[g], dtype=np.float32).reshape(-1)
        if v.size != hi - lo:
            raise ValueError(f"state.{g}: {v.size} values, need {hi - lo}")
        parts.append(v)
    prompt = None
    for v in lang.values():
        v = v[0] if isinstance(v, (list, tuple)) and v else v
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, str) and v:
            prompt = v
            break
    if prompt is None:
        raise ValueError(f"no prompt in observation['language'] ({sorted(lang)})")
    return {"images": images, "state": np.concatenate(parts).astype(np.float32), "prompt": prompt}


def reply_from_actions(actions: np.ndarray) -> dict:
    """(H, 54) absolute joint targets -> the runner's action groups, (1, H, d) float32 each."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 54:
        raise ValueError(f"actions must be (H, 54), got {a.shape}")
    return {f"action.{g}": np.ascontiguousarray(a[None, :, lo:hi]) for g, (lo, hi) in GROUPS_54.items()}


# ---------------------------------------------------------------------------------------------------- policy
class Policy:
    def __init__(self, checkpoint: pathlib.Path) -> None:
        import jax  # noqa: PLC0415
        from openpi.policies import policy_config  # noqa: PLC0415

        jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
        cfg = vega_config.make_train_config()
        norm = checkpoint / "assets" / vega_config.ASSET_ID / "norm_stats.json"
        if not (checkpoint / "params").is_dir() or not norm.is_file():
            raise SystemExit(f"{checkpoint} is not a checkpoint dir: need params/ and assets/{vega_config.ASSET_ID}/"
                             "norm_stats.json (download it from Hugging Face as in README.md)")
        t0 = time.monotonic()
        self.policy = policy_config.create_trained_policy(cfg, checkpoint)
        self.load_s = time.monotonic() - t0
        self.horizon, self.dim = int(cfg.model.action_horizon), int(cfg.model.action_dim)
        self.slots = dict(vega_config.CAMERAS.slots())
        logging.info("model %s loaded from %s in %.1f s", cfg.name, checkpoint, self.load_s)

    def act(self, inp: dict, noise: np.ndarray) -> np.ndarray:
        out = self.policy.infer(inp, noise=noise)
        a = np.asarray(out["actions"], dtype=np.float32)
        if a.shape != (self.horizon, 54):
            raise RuntimeError(f"policy returned actions {a.shape}, need ({self.horizon}, 54)")
        return a


class Server:
    def __init__(self, policy: Policy, info: dict) -> None:
        self.policy = policy
        self.contract = contract(camera_names(policy.slots), policy.horizon, info)
        self.rng = np.random.default_rng()
        self.query, self.total, self.n_reset = 0, 0, 0

    def noise(self) -> np.ndarray:
        """The flow-matching start noise of one query: N(0, 1), float32, (horizon, dim)."""
        return self.rng.standard_normal((self.policy.horizon, self.policy.dim), dtype=np.float32)

    def handle(self, ep: str, data: dict | None) -> dict:
        if ep == "ping":
            return {"status": "ok"}
        if ep == "get_modality_config":
            return self.contract
        if ep == "reset":
            self.query = 0
            self.n_reset += 1
            logging.info("reset #%d", self.n_reset)
            return {"status": "reset"}
        if ep == "get_action":
            data = data or {}
            step = (data.get("options") or {}).get("step")
            inp = parse_observation(data.get("observation"), self.policy.slots)
            t0 = time.monotonic()
            actions = self.policy.act(inp, self.noise())
            logging.info("query %d step %s %.0f ms", self.query, step, 1e3 * (time.monotonic() - t0))
            self.query += 1
            self.total += 1
            return reply_from_actions(actions)
        raise ValueError(f"unknown endpoint {ep!r}")


def warm_up(server: Server) -> float:
    """One dummy inference through the full path (parse -> transforms -> compile -> outputs) before binding."""
    obs = {"video": {n: np.zeros((1, 1, 360, 480, 3), np.uint8) for n in camera_names(server.policy.slots)},
           "state": {g: np.zeros((1, 1, hi - lo), np.float32) for g, (lo, hi) in GROUPS_54.items()},
           "language": {"annotation.human.action.task_description": [vega_config.TASK_PROMPT]}}
    t0 = time.monotonic()
    server.handle("get_action", {"observation": obs, "options": {"step": 0}})
    server.query = server.total = 0
    return time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vega-U + Wuji pipette pi0.5 policy server (ZeroMQ, inference only)")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT,
                    help="checkpoint dir with params/ and assets/ (default: $VEGA_CHECKPOINT or checkpoints/vega-wuji-pipette-pi05)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost only)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--ready-file", default=None, help="written (JSON) once the server answers requests")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [policy] %(message)s", force=True)

    policy = Policy(a.checkpoint.resolve())
    info = {"name": SERVER_NAME, "config": vega_config.CONFIG_NAME, "load_s": round(policy.load_s, 1), "pid": os.getpid()}
    server = Server(policy, info)
    warm = warm_up(server)
    info["warmup_s"] = round(warm, 2)
    logging.info("warm-up inference %.1f s", warm)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{a.host}:{a.port}")
    logging.info("listening on tcp://%s:%d", a.host, a.port)
    if a.ready_file:
        tmp = f"{a.ready_file}.tmp.{os.getpid()}"
        pathlib.Path(tmp).write_text(json.dumps({"pid": os.getpid(), "port": a.port, "contract": server.contract},
                                                default=str, indent=1))
        os.replace(tmp, a.ready_file)
    stop = {"sig": None}

    def _sig(signum: int, _frame: object) -> None:
        stop["sig"] = signum

    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, _sig)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while stop["sig"] is None:
        if not dict(poller.poll(500)):
            continue
        raw = sock.recv()
        try:
            req = unpack(raw)
            ep = req.get("endpoint", "get_action") if isinstance(req, dict) else None
            if ep == "kill":
                sock.send(pack({"status": "bye"}))
                break
            rep = server.handle(ep, req.get("data") if isinstance(req, dict) else None)
        except Exception as e:  # noqa: BLE001 -- the client raises on {"error": ...}
            logging.exception("request failed")
            rep = {"error": f"{type(e).__name__}: {e}"}
        sock.send(pack(rep))
    logging.info("exit (signal %s, %d queries, %d resets)", stop["sig"], server.total, server.n_reset)
    sock.close(0)
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
