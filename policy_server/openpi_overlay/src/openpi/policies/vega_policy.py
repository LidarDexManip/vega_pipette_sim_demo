# Vega-U + Wuji pipette policy transforms for openpi, by LIDAR (Apache License 2.0).
"""Vega-U + 2x Wuji hands (pipette task) for pi0.5: data layout, camera mapping and transforms (Vega O6, O8, clip).

Robot layout (LeRobot v2.1, 30 fps): `observation.state` and `action` are 54-D joint vectors
[L arm 7, L hand 20, R arm 7, R hand 20] (names in the dataset's meta/info.json).

Model layout, 46-D: [L arm 7, L hand 16, R arm 7, R hand 16]. The 8 dropped dims (54-D indices 12, 16, 20, 24, 39, 43,
47, 51) are the MCP abduction of the index, middle, ring and pinky fingers. They are constant 0 in every action of the
v1 set and the W16 wave; `VegaInputs` refuses any action chunk where they are not 0, and `VegaOutputs` re-inserts them
as 0. The state drops the same 8 dims (their measured values move only through contact).

Transforms, in pipeline order:
- `PromptMix` (dataset only, O8): per frame, with probability p, the stage sentence becomes the overall task sentence.
- `ClipToSoftLimits` (flag): the env's 0.95 soft-limit clip applied to stored 54-D actions. v1 and W16 labels are
  unclipped targets; the clip reproduces the executed targets exactly (CLIP_CHECK.md). v2.3 labels need no clip.
- `VegaInputs`: slots -> HWC uint8 images, 54 -> 46-D state and actions.
- `DeltaActions` / `AbsoluteActions` with `DELTA_MASK`: arms chunk-relative, hands absolute (openpi transforms).
- `VegaOutputs`: 46 -> 54-D actions (dropped dims = 0), plus the four groups the runner reads.
"""

import dataclasses
import functools
import json

import einops
import numpy as np

from openpi import transforms

# ----------------------------------------------------------------------------------------------------------- layout
DIM_54 = 54
# MCP abduction of index, middle, ring, pinky; left hand (7..26) then right hand (34..53).
DROP_DIMS = (12, 16, 20, 24, 39, 43, 47, 51)
KEEP_DIMS = tuple(i for i in range(DIM_54) if i not in DROP_DIMS)
DIM_46 = len(KEEP_DIMS)
# Arms chunk-relative (a[t+k] - s[t]), hands absolute, on the 46-D layout.
DELTA_MASK = transforms.make_bool_mask(7, -16, 7, -16)
# Batch 2: as DELTA_MASK plus the right THUMB (46-D 30..33 = 54-D 34..37: cmc_flex, cmc_abd, mcp, ip) state-relative.
# The right fingers that hold the pipette and the whole left hand stay absolute. Learnability lens (b1 labels): the
# across-episode std of the press-start mcp label is 0.149 rad absolute vs 0.0095 rad state-relative.
DELTA_MASK_THUMB = transforms.make_bool_mask(7, -16, 7, 4, -12)
# Action groups of the 54-D vector, as the eval runner reads them (meta/modality.json).
GROUPS_54 = {"left_arm": (0, 7), "left_hand": (7, 27), "right_arm": (27, 34), "right_hand": (34, 54)}

# The task sentence the policy was trained with.
VEGA_TASK_PROMPT = "Pick up the centrifuge tube from the rack and press the pipette into it."
VEGA_TASK_PROMPT_FULL = ("Pick up the centrifuge tube from the rack, press the pipette into it, then put the tube back in the rack "
                         "and hang the pipette back on its stand.")

# The model's image slots, in token order (Vega O3).
IMAGE_KEYS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# Soft joint limits of the 54 action joints (rad, float32), as the env clips the executed target:
# control_interface.py:46 np.clip(target, soft_lo, soft_hi); soft = mean -/+ 0.5 * range * 0.95 (robot_cfg.py:30),
# evaluated in float32 like Isaac Lab's warp kernel. Source: v2/report/w16/clip_check/out/limits.json (robot USD,
# matched to the Isaac Lab runtime joint table and the IK URDF); recomputed from the hard limits to the bit.
SOFT_LO = np.array(
    (
        -2.917449951171875, -0.4028499126434326, -2.917449951171875, -2.9881250858306885, -2.917449951171875,
        -1.326200008392334, -1.3156248331069946, -1.1250500679016113, -1.4294500350952148, -0.9815749526023865,
        -0.9815749526023865, -0.9815749526023865, -0.663100004196167, -0.9684748649597168, -0.9815749526023865,
        -0.9815749526023865, -0.663100004196167, -0.9684748649597168, -0.9815749526023865, -0.9815749526023865,
        -0.663100004196167, -0.9684748649597168, -0.9815749526023865, -0.9815749526023865, -0.663100004196167,
        -0.9684748649597168, -0.9815749526023865, -2.917449951171875, -1.50285005569458, -2.917449951171875,
        -2.9881250858306885, -2.917449951171875, -1.326200008392334, -1.0546249151229858, -1.1250500679016113,
        -1.4294500350952148, -0.9815749526023865, -0.9815749526023865, -0.9815749526023865, -0.663100004196167,
        -0.9684748649597168, -0.9815749526023865, -0.9815749526023865, -0.663100004196167, -0.9684748649597168,
        -0.9815749526023865, -0.9815749526023865, -0.663100004196167, -0.9684748649597168, -0.9815749526023865,
        -0.9815749526023865, -0.663100004196167, -0.9684748649597168, -0.9815749526023865,
    ),  # fmt: skip
    dtype=np.float32,
)
SOFT_HI = np.array(
    (
        2.917449951171875, 1.50285005569458, 2.917449951171875, 0.16112494468688965, 2.917449951171875,
        1.326200008392334, 1.0546249151229858, 1.2290501594543457, 0.6434500217437744, 1.5045750141143799,
        1.5045750141143799, 1.5045750141143799, 0.663100004196167, 2.015474796295166, 1.5045750141143799,
        1.5045750141143799, 0.663100004196167, 2.015474796295166, 1.5045750141143799, 1.5045750141143799,
        0.663100004196167, 2.015474796295166, 1.5045750141143799, 1.5045750141143799, 0.663100004196167,
        2.015474796295166, 1.5045750141143799, 2.917449951171875, 0.4028499126434326, 2.917449951171875,
        0.16112494468688965, 2.917449951171875, 1.326200008392334, 1.3156248331069946, 1.2290501594543457,
        0.6434500217437744, 1.5045750141143799, 1.5045750141143799, 1.5045750141143799, 0.663100004196167,
        2.015474796295166, 1.5045750141143799, 1.5045750141143799, 0.663100004196167, 2.015474796295166,
        1.5045750141143799, 1.5045750141143799, 0.663100004196167, 2.015474796295166, 1.5045750141143799,
        1.5045750141143799, 0.663100004196167, 2.015474796295166, 1.5045750141143799,
    ),  # fmt: skip
    dtype=np.float32,
)


def to_46(x: np.ndarray) -> np.ndarray:
    """54-D -> 46-D on the last axis (drops the 8 MCP-abduction dims)."""
    return np.asarray(x)[..., KEEP_DIMS]


def to_54(x: np.ndarray) -> np.ndarray:
    """46-D -> 54-D on the last axis; the dropped dims come back as 0."""
    x = np.asarray(x)
    out = np.zeros((*x.shape[:-1], DIM_54), dtype=x.dtype)
    out[..., KEEP_DIMS] = x
    return out


# ----------------------------------------------------------------------------------------------------------- cameras
@dataclasses.dataclass(frozen=True)
class VegaCameras:
    """Which dataset camera feeds each model image slot, and which slots run at `hires_size` (the cue views).

    The keys are LeRobot feature names, as the v2.3 teacher records them; the eval-side shim maps `video.<name>` to the
    same slots. Presets (each equals an `assemble_v2.py --cameras` preset, so a dataset built with that preset has
    exactly these 4 keys, and a wrong key fails at the first sample instead of loading another camera):
    - `CAMERAS_A` (the default; `--cameras A`): ego -> base_0 (224), side -> base_1 (224), left wrist (448), and the
      right wrist camera turned toward the tip (`right_wrist_a_view`, 448).
    - `CAMERAS_B` (`--cameras B`): ego -> base_0 (224), the meeting-point camera (`meet_view`) -> base_1 (448), left
      wrist (448), the unturned right wrist (224).
    - `CAMERAS_V1` (`--cameras base`): today's four cameras (v1, v2.2.1) in the A layout.
    - `CAMERAS_B1` (`--cameras v3`, batch 1): three views, all 224: ego -> base_0, left wrist, right wrist (the batch-1
      teacher's right wrist camera has camera A's aim). No base_1 slot: `base_1_rgb=None` drops the slot, so the model
      sees upstream's three slots (`image_keys()`).
    """

    base_0_rgb: str = "observation.images.ego_view"
    base_1_rgb: str | None = "observation.images.side_view"   # None: no base_1 slot
    left_wrist_0_rgb: str = "observation.images.left_wrist_view"
    right_wrist_0_rgb: str = "observation.images.right_wrist_a_view"
    # Slots at `hires_size`; the others at `lowres_size`.
    hires: tuple[str, ...] = ("left_wrist_0_rgb", "right_wrist_0_rgb")
    hires_size: int = 448
    lowres_size: int = 224

    def __post_init__(self):
        if set(self.hires) - set(self.image_keys()):
            raise ValueError(f"hires slots {self.hires} must be among {self.image_keys()}")

    def image_keys(self) -> tuple[str, ...]:
        """The model's image slots (the model config's `image_keys`): `IMAGE_KEYS` minus the slots set to None."""
        return tuple(slot for slot in IMAGE_KEYS if getattr(self, slot) is not None)

    def slots(self) -> dict[str, str]:
        """{model slot: dataset key}, in slot order."""
        return {slot: getattr(self, slot) for slot in self.image_keys()}

    def sizes(self) -> tuple[int, ...]:
        """Square input side per slot, in `image_keys()` order (the model config's `image_sizes`)."""
        return tuple(self.hires_size if slot in self.hires else self.lowres_size for slot in self.image_keys())


CAMERAS_A = VegaCameras()
CAMERAS_B = VegaCameras(
    base_1_rgb="observation.images.meet_view",
    right_wrist_0_rgb="observation.images.right_wrist_view",
    hires=("base_1_rgb", "left_wrist_0_rgb"),
)
CAMERAS_V1 = VegaCameras(right_wrist_0_rgb="observation.images.right_wrist_view")
CAMERAS_B1 = VegaCameras(base_1_rgb=None, right_wrist_0_rgb="observation.images.right_wrist_view", hires=())
# Batch 2: the batch-1 views plus the right-wrist thumb camera (teacher V2_CAM_W, key right_thumb_view; 2026-10-03:
# one extra camera on the right wrist looking back at the thumb and the plunger cap; 480x360, f 12 mm) in the base_1
# slot. All four at 224: the 7 mm plunger stroke moves the cap about 5 px there; a projected-cap probe on the 32
# re-pilot episodes reads the plunger depth at R2 0.92 / 0.40 mm (batch2/collection/check_repilot_proj.txt).
# Assemble with `assemble_v2.py --cameras ego_view,right_thumb_view,left_wrist_view,right_wrist_view --link-videos`
# (exactly the views the model reads); evaluate with P6_CAMS=v3w (renders right_thumb_view with the teacher's pose).
CAMERAS_B2 = VegaCameras(base_1_rgb="observation.images.right_thumb_view",
                         right_wrist_0_rgb="observation.images.right_wrist_view", hires=())


# -------------------------------------------------------------------------------------------------------- transforms
def _hwc_uint8(image) -> np.ndarray:
    """LeRobot video frames arrive as float CHW in [0, 1]; the runner sends uint8 HWC. Both become uint8 HWC.

    Rounds rather than truncates, so a decoded uint8 frame survives LeRobot's /255 exactly.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ClipToSoftLimits(transforms.DataTransformFn):
    """Applies the env's soft-limit clip to stored 54-D actions (no-op without actions, i.e. at inference)."""

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            data["actions"] = np.clip(actions, SOFT_LO, SOFT_HI)
        return data


@dataclasses.dataclass(frozen=True)
class VegaInputs(transforms.DataTransformFn):
    """{"images": {slot: image}, "state": [54], "actions": [H, 54], "prompt"} -> model inputs on the 46-D layout."""

    # The model's slots (`VegaCameras.image_keys()`); other slots in the input are ignored.
    image_keys: tuple[str, ...] = IMAGE_KEYS
    # [V3.0] batch 3's tail (handover / let go) commands the left fingers' MCP abduction by up to 0.03 rad on ~4 % of
    # rows; the 46-D layout has no such dims, so True drops them like the zero ones instead of refusing the chunk.
    allow_nonzero_drop: bool = False

    def __call__(self, data: dict) -> dict:
        images = {slot: _hwc_uint8(data["images"][slot]) for slot in self.image_keys}
        inputs = {
            "image": images,
            "image_mask": {slot: np.True_ for slot in self.image_keys},
            "state": to_46(np.asarray(data["state"], dtype=np.float32)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if not self.allow_nonzero_drop and np.any(actions[..., DROP_DIMS] != 0):
                raise ValueError(f"Action dims {DROP_DIMS} must be 0 to be dropped; this chunk has nonzero values.")
            inputs["actions"] = to_46(actions)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "state_drop" in data:
            inputs["state_drop"] = data["state_drop"]
        return inputs


@dataclasses.dataclass(frozen=True)
class VegaOutputs(transforms.DataTransformFn):
    """46-D actions -> 54-D (dropped dims 0), plus the runner's four action groups."""

    def __call__(self, data: dict) -> dict:
        actions = to_54(np.asarray(data["actions"][..., :DIM_46]))
        out = {"actions": actions}
        for name, (lo, hi) in GROUPS_54.items():
            out[name] = actions[..., lo:hi]
        return out


def _unit_hash(seed: int, index: int) -> float:
    """A fixed uniform number in [0, 1) per (seed, frame index) (splitmix64), independent of workers and epochs."""
    mask = (1 << 64) - 1
    x = (index * 0x9E3779B97F4A7C15 + seed * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return x / 2.0**64


@dataclasses.dataclass(frozen=True)
class PromptMix(transforms.DataTransformFn):
    """O8 (dataset only): for a fixed `prob` share of frames, chosen by a hash of (seed, frame index), the prompt
    becomes `task_prompt`; the other frames keep their own (per-frame stage) prompt. Needs the LeRobot "index"."""

    task_prompt: str
    prob: float = 0.5
    seed: int = 0
    # [V3.0] per-episode overall task: the sentence of the frame's `task_key` row (batch 2: row 18, the press task;
    # batch 3: row 23, the full-path task) when listed here; otherwise `task_prompt`. Empty = batch-1/2 behaviour.
    task_prompts: tuple[tuple[int, str], ...] = ()
    task_key: str = "annotation.human.action.task_description"

    def __call__(self, data: dict) -> dict:
        if self.prob > 0 and _unit_hash(self.seed, int(data["index"])) < self.prob:
            prompt = self.task_prompt
            if self.task_prompts and self.task_key in data:
                prompt = dict(self.task_prompts).get(int(np.asarray(data[self.task_key]).reshape(-1)[0]), prompt)
            data = {**data, "prompt": prompt}
        return data


@dataclasses.dataclass(frozen=True)
class StateDrop(transforms.DataTransformFn):
    """Dataset only: marks a fixed `prob` share of frames (hash of (seed, frame index)) with `state_drop`, which
    `TokenizePrompt` reads to leave the state out of that frame's prompt (pi0.5 with state tokens, against the
    copy-the-state shortcut). Needs the LeRobot "index"; always sets the flag so the repack can map it."""

    prob: float = 0.3
    seed: int = 1

    def __call__(self, data: dict) -> dict:
        return {**data, "state_drop": np.bool_(_unit_hash(self.seed, int(data["index"])) < self.prob)}


# [tail + pocket memory] (2026-10-12): the tail model is told
# where the tube came from. Memory = the left end effector's position (robot frame, FK of the recorded left-arm joints) at the
# grasp latch: the first row with left-hand joint sum >= 18 rad, then the last row before L_ee rises 2 mm (the scripted
# put-back's latch), from an offline table builder (not included). x and y only (z is the same +-0.5 mm in every batch-3
# episode). Written into the prompt as two numbers on 256 bins over POCKET_RANGE, so the state dropout never removes it.
# The eval runner computes the same latch and appends the same text: keep `pocket_text` in sync with its copy there.
POCKET_RANGE = ((0.17, 0.34), (0.37, 0.55))   # robot-frame x, y [m]; batch 3 spans x 0.194-0.317, y 0.389-0.531


def pocket_text(mem) -> str:
    """'Pocket: <x bin> <y bin>' for a latched L_ee position (robot frame, metres; z ignored)."""
    b = [int(np.clip(np.floor((float(v) - lo) / (hi - lo) * 256), 0, 255)) for v, (lo, hi) in zip(mem[:2], POCKET_RANGE)]
    return f"Pocket: {b[0]} {b[1]}"


@functools.lru_cache(maxsize=4)
def _pocket_table(path: str) -> dict:
    return {int(k): v["mem"] for k, v in json.load(open(path))["episodes"].items()}


@functools.lru_cache(maxsize=4)
def _pocket_rows(path: str) -> dict:
    """episode -> the frame of its grasp latch (mem_pocket.py's `row`): the eval runner appends the memory from there on."""
    return {int(k): int(v["row"]) for k, v in json.load(open(path))["episodes"].items()}


@dataclasses.dataclass(frozen=True)
class PocketMemory(transforms.DataTransformFn):
    """Dataset only: appends the episode's pocket memory (`pocket_text`) to the prompt. `table` = mem_pocket.py's json;
    an episode without an entry (batch 2, which has no tail) keeps its prompt. Needs the LeRobot "episode_index"."""

    table: str
    # [single model, 10-13] only frames at / after the episode's grasp latch get the memory (needs "frame_index"), and the
    # latched x / y get gaussian noise of this std [m] in training (the eval's latch is a few mm off the true pocket).
    after_latch: bool = False
    noise_m: float = 0.0

    def __call__(self, data: dict) -> dict:
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        mem = _pocket_table(self.table).get(ep)
        if mem is None:
            return data
        if self.after_latch and int(np.asarray(data["frame_index"]).reshape(-1)[0]) < _pocket_rows(self.table)[ep]:
            return data
        if self.noise_m > 0:
            mem = [float(v) + float(np.random.normal(0.0, self.noise_m)) for v in mem[:2]]
        return {**data, "prompt": f"{data['prompt']}. {pocket_text(mem)}"}


@dataclasses.dataclass(frozen=True)
class ForceStateDrop(transforms.DataTransformFn):
    """Eval only: every query is tokenized without the state, like the 30 % state-dropout training frames (10-12: does
    V3.1a's stall in the alignment depend on the state tokens?). Goes before VegaInputs, which passes the flag on."""

    def __call__(self, data: dict) -> dict:
        return {**data, "state_drop": np.True_}


def make_vega_example(sizes: dict[str, tuple[int, int]] | None = None) -> dict:
    """A random observation in the policy's input format (54-D state, uint8 HWC images per slot)."""
    rng = np.random.default_rng(0)
    sizes = sizes or {"base_0_rgb": (240, 424), "base_1_rgb": (360, 640)}
    return {
        "images": {
            slot: rng.integers(256, size=(*sizes.get(slot, (360, 480)), 3), dtype=np.uint8) for slot in IMAGE_KEYS
        },
        "state": to_54(rng.uniform(-1, 1, DIM_46).astype(np.float32)),
        "prompt": VEGA_TASK_PROMPT,
    }
