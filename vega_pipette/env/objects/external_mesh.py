# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Generic external-mesh prop spawner (loads one or more mesh files as a scene prop).

A small, reusable spawner for dropping an authored mesh asset (e.g. a lab centrifuge,
a tube rack) into a scene without writing a bespoke module per object. Loads anything
trimesh reads (OBJ, GLB/glTF, STL, ...; not FBX). It generalises
:mod:`tasks.common_scene.pipette_stand_object` (which is visual-only and hard-wired to
one OBJ) by adding:

  * **multi-file / multi-part assemblies** - ``obj_paths`` may list several files that share
    one authoring coordinate frame (e.g. ``Centrifuge_Tube`` ships as tube + lid + liquid +
    holder), and hierarchical files (GLB/glTF) have their per-node transforms baked in. All
    parts are oriented/scaled/placed with a single shared transform derived from their
    *combined* bounds, so relative layout is kept and the whole assembly is centred / rested
    on the table as one unit.
  * **unit + orientation handling** - Blender OBJ/GLB exports are Y-up in metres; Maya OBJ
    exports are Y-up in centimetres. Set ``up_axis`` and either ``unit_scale`` (raw ->
    metres) or ``target_height`` (fit the standing height to a length in metres).
  * **embedded textures** - ``use_embedded_textures`` binds each part's baseColor (albedo)
    map (e.g. from a GLB) as a UsdPreviewSurface instead of a flat colour; ``use_pbr_textures``
    additionally wires the normal + metallic/roughness maps when present.
  * **optional collision** - see the collision note below.

Collision note (answers "can the collider be the mesh itself?")
---------------------------------------------------------------
``collision`` selects how the physics collider is derived from the render mesh:

  * ``"off"``                 - visual only, no collider (pure background scenery).
  * ``"trimesh"``             - the **exact triangle mesh** is the collider
                                (USD approximation ``"none"``). Most accurate, and it
                                preserves concavities (rack slots, hollow bodies). PhysX
                                only allows a triangle-mesh collider on a **static or
                                kinematic** body, which is what a table prop is - so it is
                                the right default here.
  * ``"convex_hull"``         - single convex wrap (fills concavities). Cheap; the only
                                option PhysX allows for a fast-moving **dynamic** body.
  * ``"convex_decomposition"``- multiple convex pieces approximating the concave shape;
                                use this if the prop must be **dynamic** (grasped / knocked
                                about) yet keep its hollows.

So: yes, the collider can be the mesh itself (``"trimesh"``) as long as the prop stays
static/kinematic. Make it dynamic and you must fall back to a convex approximation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.sim.utils as prim_utils  # Isaac 6: prim helpers moved into isaaclab.sim.utils
import isaaclab.sim.utils as stage_utils
import numpy as np
import trimesh
from isaaclab.sim import schemas
from isaaclab.sim.spawners.meshes import meshes, meshes_cfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd, UsdShade

# map the friendly ``collision`` mode -> USD MeshCollisionAPI approximation token.
_APPROX = {
    "trimesh": "none",                       # exact triangle mesh (static/kinematic only)
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
}


def _orient_matrix(up_axis: str, flip: bool, yaw_deg: float) -> np.ndarray:
    """4x4 mapping the OBJ's ``up_axis`` onto +Z, optional 180 deg flip, then a world-Z yaw.

    Composed as ``R_yaw @ R_flip @ R_up`` so a vertex is first stood upright, then flipped
    end-for-end if requested, then spun about the vertical to orient its footprint.
    """
    up = up_axis.upper()
    if up == "Y":       # Y -> Z : +90 deg about X (Blender/Maya default OBJ export is Y-up)
        r_up = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    elif up == "X":     # X -> Z : -90 deg about Y
        r_up = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0, 1, 0])
    else:               # "Z": already vertical
        r_up = np.eye(4)
    r_flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]) if flip else np.eye(4)
    r_yaw = trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])
    return r_yaw @ r_flip @ r_up


def _load_parts(cfg: "ExternalMeshObjectCfg") -> list[tuple[str, trimesh.Trimesh, str]]:
    """Load every mesh file in ``cfg.obj_paths`` into a flat ``(part_name, mesh, src)`` list.

    For a scene file each geometry is baked into the asset frame with its **scene-graph
    node transform applied** - essential for hierarchical formats like GLB/glTF, whose node
    transforms carry per-part scale/placement (skipping them explodes the assembly); a flat
    OBJ simply has identity node transforms, so this is a no-op there.

    A single-geometry file is named after its basename (stable, controllable - OBJ ``.mtl``
    files may be missing so trimesh's material keys are unreliable); a multi-geometry file
    yields one entry per geometry, named ``"<basename>:<geomname>"``. UVs (``visual.uv``) and
    the PBR material ride along on each mesh for optional embedded texturing.
    """
    parts: list[tuple[str, trimesh.Trimesh, str]] = []
    for path in cfg.obj_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            nodes = list(loaded.graph.nodes_geometry)
            single = len(nodes) == 1
            for node_name in nodes:
                transform, geom_name = loaded.graph[node_name]
                geom = loaded.geometry[geom_name].copy()
                geom.apply_transform(transform)   # bake node transform into vertices (UVs unaffected)
                name = base if single else f"{base}:{geom_name}"
                parts.append((name, geom, path))
        else:
            parts.append((base, loaded, path))
    return parts


def _part_color(cfg: "ExternalMeshObjectCfg", name: str, mesh: trimesh.Trimesh) -> tuple:
    """Pick a flat colour for a part: explicit override, then .mtl colour, then default."""
    overrides = cfg.color_overrides or {}
    if name in overrides:
        return overrides[name]
    try:
        return tuple(float(c) for c in np.asarray(mesh.visual.material.main_color)[:3] / 255.0)
    except Exception:
        return cfg.default_color


# PBR channel -> trimesh PBRMaterial attribute holding that map's PIL image.
_PBR_ATTR = {
    "baseColor": "baseColorTexture",
    "metallicRoughness": "metallicRoughnessTexture",
    "normal": "normalTexture",
}


def _export_pbr_textures(
    mesh: trimesh.Trimesh, cache_dir: str, tag: str, src: str, want_pbr: bool
) -> dict[str, str]:
    """Bake a part's embedded PBR maps to PNGs and return ``{channel: png_path}``.

    Always attempts ``baseColor`` (albedo); when ``want_pbr`` also bakes ``metallicRoughness``
    (glTF-packed: G=roughness, B=metallic) and ``normal``. RTX samples file textures reliably
    (unlike trimesh's in-memory images), so each map is written once under ``_texcache`` and
    reused until the source file is newer. Channels the part lacks are simply omitted.
    """
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return {}
    channels = ["baseColor"] + (["metallicRoughness", "normal"] if want_pbr else [])
    out: dict[str, str] = {}
    for ch in channels:
        img = getattr(mat, _PBR_ATTR[ch], None)
        if img is None:
            continue
        os.makedirs(cache_dir, exist_ok=True)
        png = os.path.join(cache_dir, f"{tag}_{ch}.png")
        if (not os.path.exists(png)) or (os.path.exists(src) and os.path.getmtime(png) < os.path.getmtime(src)):
            img.convert("RGB").save(png)
        out[ch] = png
    return out


def _bind_pbr_material(
    stage: Usd.Stage, mesh_prim: Usd.Prim, material_root: str, uv: np.ndarray, textures: dict[str, str],
    cfg: "ExternalMeshObjectCfg",
) -> None:
    """Author + bind a UsdPreviewSurface wired to whichever PBR maps ``textures`` provides.

    Writes the mesh UVs as a shared vertex ``st`` primvar (one reader feeds every sampler) and
    connects: baseColor -> diffuseColor (sRGB); metallicRoughness -> roughness(:g)/metallic(:b)
    (raw, glTF packing); normal -> normal with the tangent-space remap scale/bias (raw). Missing
    maps fall back to the flat ``cfg.metallic``/``cfg.roughness`` scalars. glTF UVs use a
    top-left origin vs USD's bottom-left, so V is flipped when ``cfg.texture_flip_v`` is set.
    """
    # The unified RL launcher resolves configs before SimulationApp starts.
    # USD/PhysX bindings are therefore imported only when scene spawning begins.
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    st = uv.astype("float32").copy()
    if cfg.texture_flip_v:
        st[:, 1] = 1.0 - st[:, 1]
    st_pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st_pv.Set(Vt.Vec2fArray.FromNumpy(st))

    material = UsdShade.Material.Define(stage, material_root)
    shader = UsdShade.Shader.Define(stage, f"{material_root}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    st_reader = UsdShade.Shader.Define(stage, f"{material_root}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def _sampler(name: str, path: str, colorspace: str, scale: tuple[float, ...] | None = None,
                 bias: tuple[float, ...] | None = None) -> UsdShade.Shader:
        tex = UsdShade.Shader.Define(stage, f"{material_root}/{name}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(path)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if scale is not None:
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return tex

    # baseColor (albedo) -> diffuseColor
    bc = _sampler("diffuseTexture", textures["baseColor"], "sRGB")
    bc.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), "rgb")

    # metallicRoughness (glTF packs roughness in G, metallic in B) -> two scalar inputs
    if "metallicRoughness" in textures:
        mr = _sampler("mrTexture", textures["metallicRoughness"], "raw")
        mr.CreateOutput("g", Sdf.ValueTypeNames.Float)
        mr.CreateOutput("b", Sdf.ValueTypeNames.Float)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(mr.ConnectableAPI(), "g")
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(mr.ConnectableAPI(), "b")
    else:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(cfg.roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(cfg.metallic)

    # normal map -> normal, remapping the [0,1] texture to a [-1,1] tangent-space vector
    if "normal" in textures:
        nrm = _sampler(
            "normalTexture", textures["normal"], "raw",
            scale=(2.0, 2.0, 2.0, 1.0), bias=(-1.0, -1.0, -1.0, 0.0),
        )
        nrm.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm.ConnectableAPI(), "rgb")

    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)


def _apply_collision(mesh_prim_path: str, cfg: "ExternalMeshObjectCfg", stage: Usd.Stage) -> None:
    """Author the requested collider on an already-spawned visual mesh prim.

    Applies the USD collision schema and the mode's approximation token. Isaac Sim 6
    uses that standard token to select convex-decomposition cooking; no separate
    ``pxr.PhysxSchema`` API is required. Binds ``cfg.physics_material`` if given.
    """
    from pxr import UsdPhysics

    schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)
    mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
    approx = _APPROX[cfg.collision]
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).GetApproximationAttr().Set(approx)
    if cfg.physics_material is not None:
        material_path = f"{os.path.dirname(mesh_prim_path)}/physicsMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)


@clone
def spawn_external_mesh(
    prim_path: str,
    cfg: "ExternalMeshObjectCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the external-mesh prop (optionally with a collider) at ``prim_path``.

    Decorated with :func:`clone` so a regex ``prim_path`` is built once and cloned across
    environments. ``translation``/``orientation`` (from the asset ``init_state``) place the
    root Xform; the mesh is authored in the asset frame, footprint centred on the origin and
    base rested at local z = 0, so the spawn ``pos`` z should be the table surface.
    """
    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)

    parts = _load_parts(cfg)

    # 1) orient every part with one shared transform (stand up -> flip -> yaw).
    t_orient = _orient_matrix(cfg.up_axis, cfg.flip, cfg.yaw_deg)
    for _, mesh, _ in parts:
        mesh.apply_transform(t_orient)

    # 2) scale: fit the standing height to `target_height`, else apply the raw `unit_scale`;
    #    then an extra uniform `scale_multiplier`. Derived from the COMBINED bounds so a
    #    multi-file assembly is scaled as one rigid unit.
    combined = trimesh.util.concatenate([m.copy() for _, m, _ in parts])
    if cfg.target_height is not None:
        vert_extent = float(combined.extents[2])
        scale = cfg.target_height / vert_extent if vert_extent > 0.0 else 1.0
    else:
        scale = cfg.unit_scale
    scale *= cfg.scale_multiplier
    for _, mesh, _ in parts:
        mesh.apply_scale(scale)

    # 3) place: centre the combined footprint on the asset origin (unless disabled) and rest
    #    the combined base at local z = 0, then apply the xy/z nudges.
    combined = trimesh.util.concatenate([m.copy() for _, m, _ in parts])
    lo, hi = combined.bounds
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    translate = [
        (-cx if cfg.recenter_xy else 0.0) + cfg.xy_offset[0],
        (-cy if cfg.recenter_xy else 0.0) + cfg.xy_offset[1],
        (-lo[2] if cfg.base_on_ground else 0.0) + cfg.z_offset,
    ]
    for _, mesh, _ in parts:
        mesh.apply_translation(translate)

    # per-asset PNG cache for baked embedded textures (see _export_base_color_png).
    cache_dir = os.path.join(os.path.dirname(cfg.obj_paths[0]), "_texcache") if cfg.obj_paths else None

    # 4) spawn each part as its own visual mesh prim, then add the collider if requested.
    #    Visual-only prims have no rigid/collision/mass props. When `use_embedded_textures`
    #    is set and a part carries UVs + an embedded baseColor map, bind that texture instead
    #    of a flat colour; otherwise fall back to the per-part flat colour.
    for i, (name, mesh, src) in enumerate(parts):
        uv = getattr(mesh.visual, "uv", None)
        textures: dict[str, str] = {}
        if cfg.use_embedded_textures and uv is not None and len(uv) == len(mesh.vertices):
            src_base = os.path.splitext(os.path.basename(src))[0]
            textures = _export_pbr_textures(mesh, cache_dir, f"{src_base}_{i:02d}", src, cfg.use_pbr_textures)
        textured = "baseColor" in textures

        sub_cfg = meshes_cfg.MeshCuboidCfg(
            size=(1.0, 1.0, 1.0),  # unused: geometry comes from `mesh`
            rigid_props=None,
            collision_props=None,
            mass_props=None,
            physics_material=None,
            # when textured we bind our own material below, so skip the flat PreviewSurface.
            visual_material=None if textured else sim_utils.PreviewSurfaceCfg(
                diffuse_color=_part_color(cfg, name, mesh), metallic=cfg.metallic, roughness=cfg.roughness
            ),
        )
        sub_path = f"{prim_path}/Part_{i}"
        meshes._spawn_mesh_geom_from_mesh(sub_path, sub_cfg, mesh, translation=None, orientation=None)
        mesh_prim_path = f"{sub_path}/geometry/mesh"
        mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
        # keep imported polygons exactly as authored (no subdivision smoothing)
        mesh_prim.GetAttribute("subdivisionScheme").Set("none")

        if textured:
            _bind_pbr_material(
                stage, mesh_prim, f"{sub_path}/Looks/Material", np.asarray(uv), textures, cfg
            )
        if cfg.collision != "off":
            _apply_collision(mesh_prim_path, cfg, stage)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class ExternalMeshObjectCfg(SpawnerCfg):
    """Spawner config for a generic external-mesh prop. All dimensions in metres.

    A prop spawned via :class:`~isaaclab.assets.AssetBaseCfg` with ``collision != "off"``
    becomes a **static** collider (no rigid body) - the intended use for immovable table
    scenery. Wrap it in a ``RigidObjectCfg`` with kinematic rigid props if you need a
    kinematic collider instead.
    """

    func: Callable = spawn_external_mesh

    # --- source geometry ---------------------------------------------------------------
    # One or more OBJ files sharing a single authoring coordinate frame; loaded together.
    obj_paths: tuple[str, ...] = ()

    # --- orientation + units -----------------------------------------------------------
    up_axis: str = "Y"                      # OBJ axis that points up (Blender/Maya -> "Y")
    flip: bool = False                      # 180 deg flip if it spawns upside down
    yaw_deg: float = 0.0                    # spin about world +Z to orient the footprint
    # raw-units -> metres factor, used only when `target_height` is None. 1.0 = already metres,
    # 0.01 = centimetres (Maya default). Ignored if `target_height` is set.
    unit_scale: float = 1.0
    # if set, uniformly scale so the standing (Z) extent equals this many metres (overrides
    # `unit_scale`). Handy when the raw asset scale is arbitrary.
    target_height: float | None = None
    scale_multiplier: float = 1.0           # extra uniform scale on top of the above

    # --- placement (asset frame) -------------------------------------------------------
    recenter_xy: bool = True                # centre the combined footprint on the origin
    base_on_ground: bool = True             # rest the combined base at local z = 0
    xy_offset: tuple[float, float] = (0.0, 0.0)  # shift footprint off the origin [m]
    z_offset: float = 0.0                   # raise(+)/lower(-) the base off the spawn z [m]

    # --- appearance --------------------------------------------------------------------
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75)  # when no .mtl / override
    color_overrides: dict[str, tuple[float, float, float]] | None = None  # by part name
    metallic: float = 0.0
    roughness: float = 0.5
    # bind each part's embedded baseColor (albedo) texture instead of a flat colour. Works
    # for formats that carry UVs + textures (GLB/glTF, textured OBJ); parts without a texture
    # fall back to the flat colour.
    use_embedded_textures: bool = False
    # also wire the normal + metallic/roughness maps when a part has them (needs
    # `use_embedded_textures`). When off, `metallic`/`roughness` above are used as flat scalars.
    use_pbr_textures: bool = True
    texture_flip_v: bool = True   # flip UV V (glTF top-left origin vs USD bottom-left)

    # --- collision (see module docstring) ----------------------------------------------
    # "off" | "trimesh" | "convex_hull" | "convex_decomposition".
    collision: str = "off"
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
