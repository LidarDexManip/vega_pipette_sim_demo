"""Client for the policy server (policy_server/serve.py): ZeroMQ REQ/REP with msgpack messages.

Request : ``{"endpoint": str, "data": {...}}``; numpy arrays travel as ``{"__ndarray_class__": True, "as_npy": <bytes
of np.save>}``. Response: msgpack with the same array encoding, or ``{"error": str}`` on failure. The endpoints (ping,
get_modality_config, reset, get_action) are documented at the top of policy_server/serve.py.

numpy, pyzmq and msgpack only: no Isaac or torch import, so it also works from a plain Python environment.
"""
from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq

TIMEOUT_MS = 120_000
"""How long a request may wait for its reply; the server's first query can include a model compile."""


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"cannot encode type {type(obj)}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, strict_types=False)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


class PolicyClient:
    """Synchronous client: one request at a time."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = TIMEOUT_MS) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    @classmethod
    def from_url(cls, url: str, timeout_ms: int = TIMEOUT_MS) -> PolicyClient:
        """Connect to zmq://host:port and check that the server answers."""
        if not url.startswith("zmq://"):
            raise ValueError(f"policy url must be zmq://host:port, got {url!r}")
        host, port = url[len("zmq://"):].rsplit(":", 1)
        client = cls(host or "127.0.0.1", int(port), timeout_ms=timeout_ms)
        if not client.ping():
            raise RuntimeError(f"policy server at {url} does not answer")
        return client

    @staticmethod
    def to_chunk(reply: dict, groups: dict[str, dict]) -> np.ndarray:
        """A get_action reply -> (T, D) chunk, each group placed at its {start, end} columns by name."""
        parts = {}
        for key, value in reply.items():
            name = str(key).split(".", 1)[-1]          # "action.left_arm" -> "left_arm"
            if name in groups:
                value = np.asarray(value, dtype=np.float32)
                parts[name] = value[0] if value.ndim == 3 else value
        missing = [g for g in groups if g not in parts]
        if missing:
            raise KeyError(f"policy reply lacks action groups {missing}; got {sorted(reply)}")
        rows = min(len(v) for v in parts.values())
        chunk = np.zeros((rows, max(g["end"] for g in groups.values())), dtype=np.float32)
        for name, g in groups.items():
            if parts[name].shape[1] != g["end"] - g["start"]:
                raise ValueError(f"group {name}: {parts[name].shape[1]} columns, expected {g['end'] - g['start']}")
            chunk[:, g["start"]:g["end"]] = parts[name][:rows]
        return chunk

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._sock.send(pack(request))
        result = unpack(self._sock.recv())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"policy server error: {result['error']}")
        return result

    def ping(self) -> bool:
        try:
            self.call("ping")
        except (zmq.ZMQError, RuntimeError):
            return False
        return True

    def reset(self) -> dict:
        """Start a rollout (the server's query count restarts)."""
        return self.call("reset")

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config")

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        return self.call("get_action", {"observation": observation, "options": options})

    def close(self) -> None:
        self._sock.close(0)
