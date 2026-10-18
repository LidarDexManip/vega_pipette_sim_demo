"""Object layouts: each run places the props at random, the way the training demonstrations were drawn."""
from __future__ import annotations

import numpy as np

STAND_X, STAND_Y, RACK_XY = (-0.05, 0.05), (-0.05, 0.0), (-0.05, 0.05)
"""Offsets from the nominal poses, metres: the stand in x and y, the rack in both x and y."""
PIP_RAIL = (-0.02, 0.04)
"""Pipette position along the stand rail, metres."""
HOLES = ([(-2, j) for j in range(0, 6)] + [(-1, j) for j in range(-1, 5)]
         + [(0, j) for j in range(0, 6)] + [(1, j) for j in range(-1, 5)])
"""The 24 rack pockets the tube can start in."""
POCKET_PITCH = (0.0205, 0.0230)
"""Pocket spacing in the rack, metres; odd columns are shifted by half a pocket."""
MAX_TUBE_SHIFT = 0.075
"""The tube starts at most this far (metres) from its nominal place, pocket (0, 0) of the unshifted rack."""


def parse_layout(d: dict) -> dict:
    """A layout dict {stand_dxy, pip_rail, rack_dxy, hole} with typed values."""
    return {"stand_dxy": tuple(float(x) for x in d["stand_dxy"]), "pip_rail": float(d.get("pip_rail", 0.0)),
            "rack_dxy": tuple(float(x) for x in d.get("rack_dxy", (0.0, 0.0))),
            "hole": tuple(int(x) for x in d.get("hole", (0, 0)))}


def tube_shift(rack_dxy, hole) -> float:
    """How far (metres) the tube starts from its nominal place: the rack offset plus the pocket's offset in the rack."""
    i, j = hole
    off = np.array([i * POCKET_PITCH[0], (j + 0.5 * (abs(i) % 2)) * POCKET_PITCH[1]])
    return float(np.linalg.norm(np.asarray(rack_dxy) + off))


def random_layout(rng: np.random.Generator | None = None) -> dict:
    """A new random layout: a pocket for the tube, then stand and rack offsets and the pipette's place on the rail,
    drawn again until the tube is within MAX_TUBE_SHIFT of its nominal place (as for the training demonstrations)."""
    rng = rng or np.random.default_rng()
    hole = HOLES[rng.integers(len(HOLES))]
    while True:
        rack = tuple(rng.uniform(*RACK_XY, 2))
        if tube_shift(rack, hole) <= MAX_TUBE_SHIFT:
            break
    return parse_layout({"stand_dxy": (rng.uniform(*STAND_X), rng.uniform(*STAND_Y)), "pip_rail": rng.uniform(*PIP_RAIL),
                         "rack_dxy": rack, "hole": hole})
