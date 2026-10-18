# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Procedural open-top pot object.

Builds, fully in Python (no external .usd file), a wide thin-walled **open-top pot**
(a short, squat barrel with a closed bottom and an open top) sized to act as a
place target next to the piston socket:

    * a cylindrical *wall* (trimesh annulus / tube), thin relative to its radius, and
    * a thin solid *bottom* disk that closes the lower end,

unioned into a single mesh. The default outer radius is ~3x the piston barrel radius
(0.020 m -> 0.060 m) and the default height matches the socket-wall height (0.06 m), so
the pot rim sits level with the socket walls.

Collision note: the built-in mesh spawners force a *convex hull* collision
approximation for non-box/sphere meshes, which would fill the bore solid (nothing
could be placed inside the pot). This spawner instead authors a **convex
decomposition** on the mesh prim so the hollow interior is preserved for contact -
an object can actually be dropped into the pot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.sim.utils as prim_utils  # Isaac 6: prim helpers moved into isaaclab.sim.utils
import trimesh
from isaaclab.sim.spawners.meshes import meshes, meshes_cfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_pot(
    prim_path: str,
    cfg: "PotObjectCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the open-top pot mesh at ``prim_path``.

    Decorated with :func:`clone` so a regex ``prim_path`` (``/World/envs/env_.*/Pot``)
    is built once and cloned across environments. The mesh is authored in the asset
    frame with its base at local z = 0 (so the asset origin sits on the table top) and
    the rim at local z = ``cfg.height``.
    """
    from pxr import UsdPhysics

    inner_radius = cfg.outer_radius - cfg.wall_thickness

    # --- wall: thin-walled tube (annulus extruded along Z) -----------------------------
    # trimesh annulus is centred on z = 0 (spans -h/2 .. +h/2); shift up so it spans
    # 0 .. height. The wall runs the full height; the bottom disk fills the lower cap.
    wall_mesh = trimesh.creation.annulus(
        r_min=inner_radius,
        r_max=cfg.outer_radius,
        height=cfg.height,
        sections=cfg.num_segments,
    )
    wall_mesh.apply_translation((0.0, 0.0, cfg.height / 2.0))

    # --- bottom: thin solid disk that closes the base ----------------------------------
    bottom_mesh = trimesh.creation.cylinder(
        radius=cfg.outer_radius,
        height=cfg.bottom_thickness,
        sections=cfg.num_segments,
    )
    bottom_mesh.apply_translation((0.0, 0.0, cfg.bottom_thickness / 2.0))

    pot_mesh = trimesh.util.concatenate([wall_mesh, bottom_mesh])

    mesh_cfg = meshes_cfg.MeshCylinderCfg(
        radius=cfg.outer_radius,
        height=cfg.height,
        rigid_props=cfg.rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=cfg.mass),
        collision_props=cfg.collision_props,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.color, metallic=cfg.metallic),
        physics_material=cfg.physics_material,
    )
    meshes._spawn_mesh_geom_from_mesh(
        prim_path, mesh_cfg, pot_mesh, translation=translation, orientation=orientation
    )

    # --- override the collision approximation to preserve the open interior ------------
    # _spawn_mesh_geom_from_mesh defaults non-box/sphere meshes to "convexHull", which
    # would fill the pot solid. Re-author it as a convex decomposition so the cavity is
    # real and objects can be placed inside.
    mesh_prim = prim_utils.get_prim_at_path(f"{prim_path}/geometry/mesh")
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    collision_api.GetApproximationAttr().Set("convexDecomposition")

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class PotObjectCfg(SpawnerCfg):
    """Spawner config for the procedural open-top pot.

    All dimensions are in metres. Defaults give a wide, thin-walled, low pot: outer
    radius ~3x the piston barrel radius (0.020 -> 0.060) and height equal to the socket
    wall height (0.06), so the rim is flush with the socket walls.
    """

    func: Callable = spawn_pot

    # --- geometry ----------------------------------------------------------------------
    outer_radius: float = 0.060          # ~3x the piston barrel outer radius (0.020)
    wall_thickness: float = 0.004        # thin wall -> inner radius 0.056
    height: float = 0.03                 # match the socket-wall height / 2 (low pot)
    bottom_thickness: float = 0.004      # thin closed base

    num_segments: int = 64

    # --- mass --------------------------------------------------------------------------
    mass: float = 0.5

    # --- appearance --------------------------------------------------------------------
    color: tuple[float, float, float] = (0.3, 0.5, 0.9)
    metallic: float = 0.0

    # --- physics -----------------------------------------------------------------------
    # Kinematic by default: the pot is a static place target that should not be knocked
    # around. Flip kinematic_enabled to False to let it rest dynamically on the table.
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True
    )
    collision_props: sim_utils.CollisionPropertiesCfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=0.005, rest_offset=0.0
    )
    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
