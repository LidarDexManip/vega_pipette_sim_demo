"""Action-chunk scheduling for the control loop: sync, async and emulated-latency inference.

Adapted from VLAPolicyBridge chunk.py (ChunkScheduler, queue mode with latency handoff), counted in control steps.

The policy returns a chunk of `horizon` absolute joint targets planned from the observation of one control step. The
scheduler executes them one per step and asks for the next chunk:

- sync: the loop waits for the policy whenever `exec_steps` rows of the chunk have run; the new chunk starts at row 0.
- async: the query runs in a background thread while the robot keeps executing. It is sent `lead` steps before the
  current chunk has run `exec_steps` rows (counted from the row it started at). The reply arrives d steps after its
  observation and is applied at once from row d. If the current chunk runs out first, the last action is held. By
  default `lead` is the measured latency in steps + 2, so each chunk runs about `exec_steps` rows.
- latency_steps=d: the async path with a fixed latency. The query runs at once, and its chunk is applied d steps
  later from row d. The result does not depend on machine speed.

The first chunk of a rollout is always fetched synchronously and starts at row 0.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Query = Callable[[Any], np.ndarray]
"""Policy call: observation -> (T, D) chunk of absolute targets."""


@dataclass
class QueryRecord:
    """One policy query: when its observation was taken, when and from which row its chunk was applied."""

    obs_step: int
    apply_step: int = -1
    start_row: int = 0
    latency_ms: float = 0.0


@dataclass
class _Pending:
    record: QueryRecord
    chunk: np.ndarray | None = None
    due_step: int | None = None
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class ChunkScheduler:
    """Turns policy chunks into one action per control step (see the module docstring)."""

    def __init__(self, query: Query, exec_steps: int, mode: str = "sync", latency_steps: int | None = None,
                 lead: int | None = None, log: Callable[[str], None] = print) -> None:
        if mode not in ("sync", "async"):
            raise ValueError(f"inference mode {mode!r}: sync or async")
        if latency_steps is not None and (mode != "async" or latency_steps < 0):
            raise ValueError("latency_steps needs the async mode and a value >= 0")
        self.query, self.exec_steps, self.mode = query, int(exec_steps), mode
        self.latency_steps, self.fixed_lead, self.log = latency_steps, lead, log
        self.records: list[QueryRecord] = []
        self.hold_steps: list[int] = []
        self._chunk: np.ndarray | None = None
        self._row = 0
        self._start = 0
        self._last: np.ndarray | None = None
        self._pending: _Pending | None = None
        self._latencies: deque[int] = deque(maxlen=5)
        self._thread: threading.Thread | None = None
        self._queue: deque[tuple[_Pending, Any]] = deque()
        self._wake = threading.Condition()
        self._closed = False

    @property
    def lead(self) -> int:
        """Steps before the current chunk has run `exec_steps` rows at which the next async query is sent."""
        if self.fixed_lead is not None:
            return self.fixed_lead
        if self.latency_steps is not None:
            return self.latency_steps + 2
        if not self._latencies:
            return 2
        return int(math.ceil(float(np.median(self._latencies)))) + 2

    def act(self, step: int, observe: Callable[[], Any]) -> np.ndarray:
        """The action for control step `step`; `observe()` returns the observation of this step when a query needs it."""
        if self._chunk is None or (self.mode == "sync" and self._row >= self.exec_steps):
            self._fetch_now(step, observe)
        elif self.mode == "async":
            if self._pending is None and self._row - self._start >= self.exec_steps - self.lead:
                self._send(step, observe())
            self._apply_arrival(step)
        if self._row < len(self._chunk):
            action = self._chunk[self._row]
        else:
            action = self._last
            if not self.hold_steps or self.hold_steps[-1] != step - 1:
                self.log(f"[rt   ] step {step}: chunk ran out before the next one arrived; holding the last action")
            self.hold_steps.append(step)
        self._row += 1
        self._last = action
        return action

    def close(self) -> None:
        """Stop the background thread (async mode)."""
        with self._wake:
            self._closed = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ---- internals ------------------------------------------------------------------------------------------------
    def _fetch_now(self, step: int, observe: Callable[[], Any]) -> None:
        rec = QueryRecord(obs_step=step, apply_step=step, start_row=0)
        t0 = _now()
        self._chunk = self._checked(self.query(observe()))
        rec.latency_ms = 1e3 * (_now() - t0)
        self._row = self._start = 0
        self.records.append(rec)

    def _send(self, step: int, obs: Any) -> None:
        pending = _Pending(QueryRecord(obs_step=step))
        if self.latency_steps is not None:          # emulated latency: run now, apply at step + d
            t0 = _now()
            pending.chunk = self._checked(self.query(obs))
            pending.record.latency_ms = 1e3 * (_now() - t0)
            pending.due_step = step + self.latency_steps
            pending.done.set()
        else:
            self._start_thread()
            with self._wake:
                self._queue.append((pending, obs))
                self._wake.notify_all()
        self._pending = pending

    def _apply_arrival(self, step: int) -> None:
        p = self._pending
        if p is None or not p.done.is_set() or (p.due_step is not None and step < p.due_step):
            return
        if p.error is not None:
            raise RuntimeError("policy query failed") from p.error
        d = step - p.record.obs_step
        p.record.apply_step, p.record.start_row = step, d
        self._chunk, self._row, self._start, self._pending = p.chunk, d, d, None
        self.records.append(p.record)
        self._latencies.append(d)
        self.log(f"[rt   ] query {len(self.records) - 1}: obs step {p.record.obs_step}, applied at step {step} from row "
                 f"{d} (latency {p.record.latency_ms:.0f} ms = {d} steps); lead {self.lead}")

    def _start_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="policy-query", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                pending, obs = self._queue.popleft()
            t0 = _now()
            try:
                pending.chunk = self._checked(self.query(obs))
            except BaseException as e:  # noqa: BLE001 - re-raised in the control loop
                pending.error = e
            pending.record.latency_ms = 1e3 * (_now() - t0)
            pending.done.set()

    def _checked(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or len(chunk) < self.exec_steps:
            raise RuntimeError(f"policy returned a {chunk.shape} chunk; need at least {self.exec_steps} rows")
        return chunk


def _now() -> float:
    return time.perf_counter()
