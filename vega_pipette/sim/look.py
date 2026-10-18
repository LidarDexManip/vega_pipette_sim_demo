"""Render settings of the recorded training data, applied to the one-env scene.

  - Kit arg --/renderer/scenePartitioning/enabled=true (KIT_ARG).
  - The room's ceiling RectLight is split into 6 x 10 tiles right after Isaac Lab clones the scene (install).
  - The RTPT light cache is off (apply_carb, after the env is built and before the first frame).
  - Scene partitions: env 0's objects (all but the room and the packing table) carry the partition primvar env_0, and
    every camera under the env root the token env_0 (install, on the renderer's first prepare_stage).
check() confirms all of it after the first render. No Kit or pxr import at module level.
"""
from __future__ import annotations

import collections
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Sdf, Usd

KIT_ARG = "--/renderer/scenePartitioning/enabled=true"
PRIMVAR = "primvars:omni:scenePartition"
CAM_ATTR = "omni:scenePartition"
ENV_FMT = "/World/envs/env_{}"
EXCLUDE = ("Room", "PackingTable")
SPLIT_LIGHTS = (("Room/RectLight", 6, 10),)
CARB = (("/rtx/rtpt/lightcache/cached/enabled", False),)
REPORT: dict = {}


def _set_token(layer: Sdf.Layer, path: Sdf.Path, name: str, value: str) -> None:
    """Author a uniform token on `path` in `layer` (Sdf level); creates an over if needed."""
    from pxr import Sdf
    attr_path = path.AppendProperty(name)
    spec = layer.GetAttributeAtPath(attr_path)
    if spec is None:
        if layer.GetPrimAtPath(path) is None:
            Sdf.CreatePrimInLayer(layer, path)
        Sdf.JustCreatePrimAttributeInLayer(layer, attr_path, Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform, True)
        spec = layer.GetAttributeAtPath(attr_path)
    spec.default = value


def author_partitions(stage: Usd.Stage, num_envs: int, layer: Sdf.Layer | None = None) -> dict:
    """Tag env roots' children (except EXCLUDE) with their partition primvar and cameras with the partition token;
    the excluded children of env 1.. are hidden (env 0's copy serves all)."""
    from pxr import Sdf, Usd, UsdGeom
    layer = layer or stage.GetRootLayer()
    n = collections.Counter()
    missing, hidden, tagged = [], [], []
    with Sdf.ChangeBlock():
        for e in range(int(num_envs)):
            root = stage.GetPrimAtPath(ENV_FMT.format(e))
            if not root or not root.IsValid():
                missing.append(e)
                continue
            tok = f"env_{e}"
            n["env_roots"] += 1
            for ch in root.GetChildren():
                if ch.GetName().startswith(EXCLUDE):
                    n["excluded_children"] += 1
                    if e > 0:
                        hidden.append(ch.GetPath())
                    continue
                _set_token(layer, ch.GetPath(), PRIMVAR, tok)
                n["children"] += 1
                if e == 0:
                    tagged.append(ch.GetName())
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.IsInstanceProxy():
                    continue
                if prim.IsA(UsdGeom.Camera):
                    _set_token(layer, prim.GetPath(), CAM_ATTR, tok)
                    n["cameras"] += 1
    nh = 0
    for path in hidden:
        it = iter(Usd.PrimRange(stage.GetPrimAtPath(path)))
        for q in it:
            if q.IsInstance() or q.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(q).MakeInvisible()
                nh += 1
                it.PruneChildren()
    n["hidden"] = nh
    return dict(counts=dict(n), num_envs=int(num_envs), missing_env_roots=missing, tagged_env0=tagged)


def split_lights(stage: Usd.Stage) -> list:
    """Replace each <env>/REL RectLight of SPLIT_LIGHTS by NX x NY tiles with the same attributes."""
    from pxr import Gf, Usd, UsdGeom, UsdLux
    envs = [c for c in stage.GetPrimAtPath("/World/envs").GetChildren() if c.GetName().startswith("env_")]
    rep = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for rel, nx, ny in SPLIT_LIGHTS:
            for envp in envs:
                src = stage.GetPrimAtPath(f"{envp.GetPath()}/{rel}")
                if not src or not src.IsValid() or src.GetTypeName() != "RectLight":
                    raise RuntimeError(f"look: no RectLight at {envp.GetPath()}/{rel}")
                W = float(src.GetAttribute("inputs:width").Get())
                H = float(src.GetAttribute("inputs:height").Get())
                M = UsdGeom.Xformable(src).GetLocalTransformation()
                made = []
                for i in range(nx):
                    for j in range(ny):
                        dst_path = src.GetPath().GetParentPath().AppendChild(f"{src.GetName()}_t{i}_{j}")
                        dst = UsdLux.RectLight.Define(stage, dst_path).GetPrim()
                        for sch in src.GetAppliedSchemas():
                            if sch not in dst.GetAppliedSchemas():
                                dst.AddAppliedSchema(sch)
                        for a in src.GetAttributes():
                            nm = a.GetName()
                            if nm.startswith("xformOp") or not a.HasAuthoredValue() or a.Get() is None:
                                continue
                            dst.CreateAttribute(nm, a.GetTypeName(), a.IsCustom()).Set(a.Get())
                        for nm, v in (("inputs:width", W / nx), ("inputs:height", H / ny), ("width", W / nx),
                                      ("height", H / ny)):
                            if dst.GetAttribute(nm):
                                dst.GetAttribute(nm).Set(v)
                        _toff = Gf.Matrix4d().SetTranslate(Gf.Vec3d((i + 0.5) * W / nx - W / 2,
                                                                    (j + 0.5) * H / ny - H / 2, 0.0))
                        xf = UsdGeom.Xformable(dst)
                        xf.ClearXformOpOrder()
                        xf.AddTransformOp().Set(_toff * M)
                        made.append(str(dst_path))
                src.SetActive(False)
                rep.append(dict(src=str(src.GetPath()), W=W, H=H, tiles=len(made), first=made[:2]))
    return rep


def install(log: Callable[[str], None] = print, scene_cls: type | None = None) -> None:
    """Before the environment is built: hook the light split into the scene clone and the partitions into prepare_stage (once each)."""
    if scene_cls is None:
        from isaaclab.scene.interactive_scene import (
            InteractiveScene as scene_cls,  # noqa: PLC0415
        )
    from isaaclab_physx.renderers import isaac_rtx_renderer as m  # noqa: PLC0415

    real_clone = scene_cls.clone_environments
    if getattr(real_clone, "_vega_look", False):
        raise RuntimeError("look: clone hook already installed")

    def clone_then_split(self: object, copy_from_source: bool = False) -> None:
        real_clone(self, copy_from_source=copy_from_source)
        if clone_then_split.done:
            return
        clone_then_split.done = True
        REPORT["split_light"] = split_lights(self.stage)
        log(f"[look ] split {REPORT['split_light'][:1]}")

    clone_then_split.done = False
    clone_then_split._vega_look = True
    scene_cls.clone_environments = clone_then_split

    cls = m.IsaacRtxRenderer
    orig = cls.prepare_stage

    def prepare_stage(self: object, stage: Usd.Stage, num_envs: int) -> None:
        orig(self, stage, num_envs)
        REPORT["prepare_stage_calls"] = REPORT.get("prepare_stage_calls", 0) + 1
        if REPORT["prepare_stage_calls"] > 1:
            return
        REPORT["partitions"] = author_partitions(stage, num_envs)
        log(f"[look ] partitions: {REPORT['partitions']}")

    prepare_stage._vega_look = True
    cls.prepare_stage = prepare_stage


def apply_carb(log: Callable[[str], None] = print) -> dict:
    """Apply the carb render settings: after the environment is built, before the reset and the first frame."""
    import carb.settings  # noqa: PLC0415
    s = carb.settings.get_settings()
    for k, v in CARB:
        s.set_bool(k, bool(v))
    REPORT["carb_set"] = {k: s.get(k) for k, _ in CARB}
    log(f"[look ] carb {REPORT['carb_set']}")
    return REPORT["carb_set"]


def check(num_cameras: int) -> list:
    """The render settings that must hold after the first render; returns the problems (empty: all in place)."""
    bad = []
    if not REPORT.get("split_light") or REPORT["split_light"][0].get("tiles") != 60:
        bad.append(f"split light: {REPORT.get('split_light')}")
    p = REPORT.get("partitions") or {}
    c = p.get("counts") or {}
    if c.get("env_roots") != 1 or c.get("cameras", 0) < num_cameras:
        bad.append(f"partitions: {p}")
    if REPORT.get("carb_set") != {k: v for k, v in CARB}:
        bad.append(f"carb: {REPORT.get('carb_set')}")
    return bad


VIEWPORT_CAMERA = "/OmniverseKit_Persp"


def show_viewport(eye: tuple, lookat: tuple) -> None:
    """Window mode: frame the workspace and tag the viewport camera with env 0's scene partition, so it sees what the
    policy cameras see (env 0's objects render only for cameras tagged env_0)."""
    import omni.usd  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(VIEWPORT_CAMERA)
    if prim:
        _set_token(stage.GetSessionLayer(), prim.GetPath(), CAM_ATTR, "env_0")
    SimulationContext.instance().set_camera_view(tuple(eye), tuple(lookat))
