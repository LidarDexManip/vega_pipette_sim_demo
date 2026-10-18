# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""External pipette-stand prop, spawned from an OBJ file for visualization.

Loads ``pipetteStandVertical.obj`` (a two-material Blender export: a light "weiss"
body + a small dark-blue accent) and stands it upright on the table as a static
decoration near the piston.

The OBJ is authored Y-up (Blender's default OBJ export) with its wide base at the
Y minimum; :func:`spawn_pipette_stand` rotates that vertical axis onto +Z, scales the
whole thing to ``cfg.height`` metres, centres the footprint on the asset origin and
rests the base at local z = 0 (so the spawn ``pos`` z should be the table surface).

Each OBJ material becomes its own visual mesh prim under the root so the body/accent
keep their distinct flat colours (read from the .mtl via trimesh, falling back to
``cfg.default_color``). By default each sub-mesh also gets an exact triangle-mesh
collider that follows it (``cfg.collision``); since the prop is spawned with no rigid
body it is a **static** collider, for which PhysX allows a triangle mesh. Set
``cfg.collision = "off"`` to return it to a pure visual-only prop.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

from ..constants import LAB_PROPS_DIR

if TYPE_CHECKING:
    from pxr import Usd

_DEFAULT_STAND_OBJ = str(LAB_PROPS_DIR / "pipetteStandVertical.obj")

# map the friendly ``collision`` mode -> USD MeshCollisionAPI approximation token.
# "convex_decomposition" is intentionally absent: PhysX's runtime cooking of that
# token does not survive the Newton import (the solver ends up colliding against
# whole-part convex hulls, which fill the rail groove) -- that mode is handled by
# pre-baking a CoACD decomposition instead, see :func:`_author_coacd_colliders`.
_APPROX = {
    "trimesh": "none",                       # exact triangle mesh (static/kinematic only)
    "convex_hull": "convexHull",
}


def _apply_collision(
    mesh_prim_path: str,
    cfg: "PipetteStandObjectCfg",
    stage: Usd.Stage,
    physics_material: sim_utils.RigidBodyMaterialCfg | None,
    approx: str | None = None,
) -> None:
    """Author the requested collider on an already-spawned mesh prim.

    Applies the USD collision schema + the approximation token (from ``cfg.collision``
    unless ``approx`` overrides it; ``"trimesh"`` gives an exact triangle-mesh collider
    that follows the visual mesh), and binds ``physics_material``.  Spawned via
    ``AssetBaseCfg`` with no rigid body, so this is a **static** collider - for which
    PhysX allows a triangle mesh.
    """
    import isaaclab.sim.utils as prim_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)
    mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
    if approx is None:
        approx = _APPROX[cfg.collision]
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).GetApproximationAttr().Set(approx)
    if physics_material is not None:
        material_path = f"{os.path.dirname(mesh_prim_path)}/physicsMaterial"
        physics_material.func(material_path, physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)


def _decompose_submeshes(obj_path: str, submeshes: list[trimesh.Trimesh], threshold: float) -> list:
    """CoACD pieces per submesh, in the raw OBJ frame, cached next to the OBJ.

    Returns ``pieces[i] = [trimesh.Trimesh, ...]`` for submesh ``i``.  The
    decomposition runs once (~10 s for the stand) and is cached in an ``.npz``
    keyed by the threshold; delete the cache after editing the OBJ.
    """
    cache = f"{obj_path}.coacd_t{threshold:g}.npz"
    try:
        data = np.load(cache)
        counts = data["counts"].astype(int)
        if len(counts) == len(submeshes):
            return [
                [
                    trimesh.Trimesh(vertices=data[f"v_{i}_{j}"], faces=data[f"f_{i}_{j}"], process=False)
                    for j in range(counts[i])
                ]
                for i in range(len(counts))
            ]
    except (OSError, KeyError, ValueError):
        pass

    import coacd

    coacd.set_log_level("error")
    pieces, arrays = [], {}
    for i, sub in enumerate(submeshes):
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(sub.vertices), np.asarray(sub.faces)), threshold=threshold
        )
        pieces.append([trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts])
        for j, (v, f) in enumerate(parts):
            arrays[f"v_{i}_{j}"] = np.asarray(v)
            arrays[f"f_{i}_{j}"] = np.asarray(f)
    arrays["counts"] = np.array([len(p) for p in pieces])
    try:
        np.savez_compressed(cache, **arrays)
    except OSError:
        pass
    return pieces


def _upright_rotation(up_axis: str) -> np.ndarray:
    """4x4 that maps the OBJ's vertical ``up_axis`` onto +Z (base -> -Z)."""
    up = up_axis.upper()
    if up == "Y":       # Y -> Z : +90 deg about X (Blender's default OBJ export is Y-up)
        return trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    if up == "X":       # X -> Z : -90 deg about Y
        return trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0, 1, 0])
    return np.eye(4)    # "Z": already vertical


def _resize_mesh_about(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    scale: np.ndarray,
) -> None:
    """Resize a mesh around a shared center in the upright frame.

    The stand OBJ uses local Y for its left/right span and local Z for height
    after :func:`_upright_rotation`.  Applying one shared affine transform to
    the visual rail and every CoACD piece keeps their collision geometry
    exactly aligned.
    """
    mesh.vertices = (np.asarray(mesh.vertices) - center) * scale + center


def spawn_pipette_stand(
    prim_path: str,
    cfg: "PipetteStandObjectCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Lazily apply Isaac Lab's cloning decorator after SimulationApp starts.

    Importing ``isaaclab.sim.utils`` before Kit starts pulls in ``pxr`` and can
    make Isaac Sim 6 abort during protobuf initialization.  Environment config
    modules are parsed before AppLauncher, so the custom spawner must defer that
    import until scene construction.
    """
    from isaaclab.sim.utils import clone

    return clone(_spawn_pipette_stand)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


def _spawn_pipette_stand(
    prim_path: str,
    cfg: "PipetteStandObjectCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the (visual-only) pipette stand at ``prim_path``.

    Called through the lazily-applied :func:`clone` wrapper so a regex
    ``prim_path`` (``/World/envs/env_.*/PipetteStand``) is built once and cloned
    across environments. ``translation``/``orientation`` (from the asset
    ``init_state``) place the root Xform; all mesh geometry is authored in the
    asset frame with the base at local z = 0.
    """
    import isaaclab.sim.utils as prim_utils
    import isaaclab.sim.utils as stage_utils
    from isaaclab.sim.spawners.meshes import meshes, meshes_cfg
    from pxr import UsdGeom

    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    rail_material = cfg.rail_physics_material

    # Load once merged (to derive a single shared transform) and once per-material (to keep
    # the body/accent colours). Both share the same baked coordinate frame.
    scene = trimesh.load(cfg.obj_path)
    submeshes = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]
    merged = trimesh.load(cfg.obj_path, force="mesh")

    # Build the shared orient->scale->place transform from the merged bounds so every
    # sub-mesh moves identically and stays aligned.
    rot = _upright_rotation(cfg.up_axis)
    flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]) if cfg.flip else np.eye(4)

    probe = merged.copy()
    probe.apply_transform(rot)
    probe.apply_transform(flip)
    long_extent = float(probe.extents[2])
    scale = cfg.height / long_extent if long_extent > 0.0 else 1.0
    probe.apply_scale(scale)
    stand_axis_scale = np.ones(3, dtype=float)
    if cfg.width is not None and probe.extents[1] > 0.0:
        stand_axis_scale[1] = cfg.width / float(probe.extents[1])
        _resize_mesh_about(probe, np.zeros(3, dtype=float), stand_axis_scale)
    lo, hi = probe.bounds
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    base_z = lo[2]
    translate = [
        -cx + cfg.xy_offset[0],
        -cy + cfg.xy_offset[1],
        -base_z + cfg.z_offset,
    ]

    # pre-baked CoACD decomposition (raw OBJ frame; must run before the in-place
    # transforms below).  PhysX's runtime convexDecomposition cooking is NOT used:
    # it does not survive the Newton import -- the solver ends up colliding against
    # whole-part convex hulls, which fill the rail groove the pipette hangs in.
    raw_pieces = (
        _decompose_submeshes(cfg.obj_path, submeshes, cfg.coacd_threshold)
        if cfg.collision == "convex_decomposition"
        else None
    )

    for i, geom in enumerate(submeshes):
        # Resolve the material before transforming the mesh: the dark-blue
        # material identifies the cross-rail that receives its own measured
        # dimensions below.
        try:
            mat_name = str(geom.visual.material.name).lower()
        except Exception:
            mat_name = ""
        is_rail = bool(cfg.rail_material_key) and cfg.rail_material_key in mat_name

        geom.apply_transform(rot)
        geom.apply_transform(flip)
        geom.apply_scale(scale)
        _resize_mesh_about(geom, np.zeros(3, dtype=float), stand_axis_scale)
        rail_center = None
        rail_scale = None
        if is_rail and (cfg.rail_length is not None or cfg.rail_height is not None):
            rail_lo, rail_hi = geom.bounds
            rail_extent = rail_hi - rail_lo
            rail_center = (rail_lo + rail_hi) / 2.0
            rail_scale = np.ones(3, dtype=float)
            if cfg.rail_length is not None and rail_extent[1] > 0.0:
                rail_scale[1] = cfg.rail_length / rail_extent[1]
            if cfg.rail_height is not None and rail_extent[2] > 0.0:
                rail_scale[2] = cfg.rail_height / rail_extent[2]
            _resize_mesh_about(geom, rail_center, rail_scale)
        if is_rail and cfg.rail_z_offset != 0.0:
            geom.apply_translation((0.0, 0.0, cfg.rail_z_offset))
        geom.apply_translation(translate)

        # per-material flat colour (from the .mtl), falling back to the default
        try:
            color = tuple(float(c) for c in np.asarray(geom.visual.material.main_color)[:3] / 255.0)
        except Exception:
            color = cfg.default_color
        # visual mesh: no rigid/mass props (the collider, if any, is authored below - see
        # meshes._spawn_mesh_geom_from_mesh, which gates the collider on collision_props).
        sub_cfg = meshes_cfg.MeshCuboidCfg(
            size=(1.0, 1.0, 1.0),  # unused: geometry comes from `geom`
            rigid_props=None,
            collision_props=None,
            mass_props=None,
            physics_material=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=cfg.metallic, roughness=cfg.roughness
            ),
        )
        sub_path = f"{prim_path}/Part_{i}"
        meshes._spawn_mesh_geom_from_mesh(sub_path, sub_cfg, geom, translation=None, orientation=None)
        mesh_prim_path = f"{sub_path}/geometry/mesh"
        # keep imported polygons exactly as authored (no subdivision smoothing)
        prim_utils.get_prim_at_path(mesh_prim_path).GetAttribute("subdivisionScheme").Set("none")
        # add a (static) collider, unless disabled
        if cfg.collision != "off":
            material = (
                rail_material
                if is_rail and cfg.rail_physics_material is not None
                else cfg.physics_material
            )
            if raw_pieces is not None:
                # one invisible convexHull collider per CoACD piece; the visual
                # sub-mesh itself stays collider-free so no whole-part hull exists
                for j, piece in enumerate(raw_pieces[i]):
                    pc = piece.copy()
                    pc.apply_transform(rot)
                    pc.apply_transform(flip)
                    pc.apply_scale(scale)
                    _resize_mesh_about(pc, np.zeros(3, dtype=float), stand_axis_scale)
                    if is_rail and rail_center is not None and rail_scale is not None:
                        _resize_mesh_about(pc, rail_center, rail_scale)
                    if is_rail and cfg.rail_z_offset != 0.0:
                        pc.apply_translation((0.0, 0.0, cfg.rail_z_offset))
                    pc.apply_translation(translate)
                    col_cfg = meshes_cfg.MeshCuboidCfg(
                        size=(1.0, 1.0, 1.0),  # unused: geometry comes from `pc`
                        rigid_props=None,
                        collision_props=None,
                        mass_props=None,
                        physics_material=None,
                        visual_material=None,
                    )
                    col_path = f"{sub_path}/coacd_{j}"
                    meshes._spawn_mesh_geom_from_mesh(col_path, col_cfg, pc, translation=None, orientation=None)
                    _apply_collision(f"{col_path}/geometry/mesh", cfg, stage, material, approx="convexHull")
                    UsdGeom.Imageable(prim_utils.get_prim_at_path(col_path)).MakeInvisible()
            else:
                _apply_collision(mesh_prim_path, cfg, stage, material)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class PipetteStandObjectCfg(SpawnerCfg):
    """Spawner config for the pipette stand. All dimensions in metres.

    Spawned via :class:`~isaaclab.assets.AssetBaseCfg` with ``collision != "off"`` it becomes
    a **static** collider (no rigid body) - the intended use for an immovable table prop.
    """

    func: Callable = spawn_pipette_stand

    obj_path: str = _DEFAULT_STAND_OBJ
    up_axis: str = "Y"                      # OBJ axis that points up (Blender OBJ export -> Y)
    flip: bool = False                      # 180 deg flip if it spawns upside down
    height: float = 0.285                   # calibrated overall height (base -> top) [m]
    width: float | None = None              # overall left/right outside span [m]
    rail_height: float | None = None        # blue cross-rail top/bottom height [m]
    rail_length: float | None = None        # blue cross-rail left/right span [m]
    rail_z_offset: float = 0.0              # shift only the blue rail vertically [m]
    xy_offset: tuple[float, float] = (0.0, 0.0)  # shift footprint off the asset origin [m]
    z_offset: float = 0.0                   # raise(+)/lower(-) the base off the spawn z [m]

    # appearance (used when a material colour cannot be read from the .mtl)
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    metallic: float = 0.0
    roughness: float = 0.5

    # --- collision ---------------------------------------------------------------------
    # "off" (visual only) | "trimesh" (exact mesh, static/kinematic only) | "convex_hull" |
    # "convex_decomposition" (pre-baked CoACD: one invisible convexHull collider prim per
    # piece, cached next to the OBJ -- survives the Newton import, unlike PhysX's runtime
    # cooking). Default: an exact triangle-mesh collider following the mesh.
    collision: str = "trimesh"
    coacd_threshold: float = 0.05           # CoACD concavity threshold (smaller = more pieces)
    collision_props: sim_utils.CollisionPropertiesCfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=0.005, rest_offset=0.0
    )
    physics_material: sim_utils.RigidBodyMaterialCfg | None = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    # The dark-blue OBJ material ("blue_d") is the top cross-rail the pipette hangs on;
    # it gets its own grippier material so the passively-hung pipette stays put.
    rail_material_key: str = "blue"
    rail_physics_material: sim_utils.RigidBodyMaterialCfg | None = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )
