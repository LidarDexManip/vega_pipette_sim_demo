"""The tube rack of the pick-and-place task, generated from its pocket grid (progress/rack.py).

Visual mesh and collider come from one geometry: two perforated slabs (rack.SLABS) with every pocket open, each pocket
a 16-sided bore of flat radius rack.HOLE_R with the bevel rack.BEVEL at the top of both slabs, and four visual corner
posts joining the slabs. The collider is a set of convex pieces: for every pocket its Voronoi cell (bisectors to all
other pockets) clipped to the plate, cut into 16 sectors at the bore polygon's corners; each piece is the sector between
the bore wall (and the bevel) and the cell's edge, an invisible mesh with a convex-hull collider. The rack is a heavy
dynamic rigid body.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.sim.utils as prim_utils
import numpy as np
import trimesh
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from scipy.spatial import ConvexHull

from ...progress import rack

if TYPE_CHECKING:
    from pxr import Usd

POST_SIZE = 0.006


def _clip(poly: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """The part of the convex polygon `poly` (k, 2) with n . p >= d (one half-plane of Sutherland-Hodgman)."""
    out = []
    for a in range(len(poly)):
        p, q = poly[a], poly[(a + 1) % len(poly)]
        fp, fq = n @ p - d, n @ q - d
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + fp / (fp - fq) * (q - p))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def collider_pieces() -> list[dict]:
    """The convex pieces of both slabs (module docstring): dicts of points (k, 3) and, for sorting the visual faces, the
    pocket centre, the sector's flat normal and the bore radius."""
    px, py = rack.PLATE_HALF
    plate = np.array([[-px, -py], [px, -py], [px, py], [-px, py]], float)
    r_in, (w, h), sides = rack.HOLE_R, rack.BEVEL, rack.HOLE_SIDES
    centres = {hole: rack.hole_xy(hole) for hole in rack.POCKETS}
    pieces = []
    for hole, cen in centres.items():
        cell = plate.copy()
        for other, o in centres.items():
            if other != hole:
                n = (cen - o) / np.linalg.norm(cen - o)
                cell = _clip(cell, n, n @ (0.5 * (cen + o)))
        for k in range(sides):
            a0 = 2 * math.pi * k / sides - math.pi / sides          # the sector between two polygon corners; its flat
            a1 = a0 + 2 * math.pi / sides                           # faces angle 2 pi k / sides
            sec = _clip(cell - cen, np.array([-math.sin(a0), math.cos(a0)]), 0.0)
            sec = _clip(sec, np.array([math.sin(a1), -math.cos(a1)]), 0.0) if len(sec) else sec
            if not len(sec):
                continue
            nk = np.array([math.cos(2 * math.pi * k / sides), math.sin(2 * math.pi * k / sides)])
            for z0, z1 in rack.SLABS:
                lo, hi = _clip(sec, nk, r_in), _clip(sec, nk, r_in + w)
                if not len(lo) or not len(hi):
                    continue
                pts = np.concatenate([np.c_[lo + cen, np.full(len(lo), z0)], np.c_[lo + cen, np.full(len(lo), z1 - h)],
                                      np.c_[hi + cen, np.full(len(hi), z1)]])
                pieces.append({"points": pts, "cen": cen, "nk": nk, "r_in": r_in})
    return pieces


def _hull_faces(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The convex hull's triangles (counter-clockwise seen from outside) with their outward normals."""
    hull = ConvexHull(points)
    faces = []
    for simp, eq in zip(hull.simplices, hull.equations, strict=True):
        a, b, c = points[simp]
        if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:
            simp = simp[[0, 2, 1]]
        faces.append((simp, eq[:3]))
    return faces


def visual_surfaces(pieces: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Triangles of the slabs' visible surfaces (tops, bevels, bore walls, outer sides and undersides); the faces inside
    the decomposition (sector rays, cell bisectors) are dropped. -> [(points (m, 3), faces (t, 3))] per surface kind."""
    px, py = rack.PLATE_HALF
    acc: dict[str, list] = {k: [] for k in ("top", "rim", "wall", "body")}
    for pc in pieces:
        pts = pc["points"]
        for simp, n in _hull_faces(pts):
            tri = pts[simp]
            ctr = tri.mean(0)
            if n[2] > 0.999:
                kind = "top"
            elif n[2] < -0.999:
                kind = "body"
            elif abs(n[2]) < 1e-3:
                on_edge = abs(abs(ctr[0]) - px) < 1e-6 or abs(abs(ctr[1]) - py) < 1e-6
                if abs(np.dot(ctr[:2] - pc["cen"], pc["nk"]) - pc["r_in"]) < 1e-6 and np.dot(n[:2], pc["nk"]) < -0.99:
                    kind = "wall"
                elif on_edge:
                    kind = "body"
                else:
                    continue
            else:
                kind = "rim"
            acc[kind].extend(tri)
    out = []
    for tris in acc.values():
        if tris:
            out.append((np.array(tris), np.arange(len(tris), dtype=np.int64).reshape(-1, 3)))
    return out


def posts() -> tuple[np.ndarray, np.ndarray]:
    """The four corner posts between the slabs."""
    px, py = rack.PLATE_HALF
    z0, z1 = rack.SLABS[0][1], rack.SLABS[-1][0]
    boxes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            b = trimesh.creation.box(extents=(POST_SIZE, POST_SIZE, z1 - z0))
            b.apply_translation((sx * (px - POST_SIZE / 2), sy * (py - POST_SIZE / 2), 0.5 * (z0 + z1)))
            boxes.append(b)
    m = trimesh.util.concatenate(boxes)
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)


def _mesh(stage: Usd.Stage, path: str, points: np.ndarray, faces: np.ndarray) -> Usd.Prim:
    from pxr import Gf, UsdGeom, Vt
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(x) for x in p)) for p in points]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(x) for x in np.asarray(faces).reshape(-1)]))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(x) for x in points.min(0))),
                                      Gf.Vec3f(*(float(x) for x in points.max(0)))]))
    return m.GetPrim()


@clone
def spawn_tube_rack(prim_path: str, cfg: TubeRackCfg, translation: tuple[float, float, float] | None = None,
                    orientation: tuple[float, float, float, float] | None = None, **kwargs) -> Usd.Prim:
    """Author the rack at prim_path: the visual surfaces with one colour, the collider pieces with one physics
    material, then the rigid body and its mass."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    stage = prim_utils.get_current_stage()
    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    pieces = collider_pieces()

    look = UsdShade.Material.Define(stage, f"{prim_path}/Visual/Looks/Rack")
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Visual/Looks/Rack/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) for c in cfg.color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(cfg.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for k, (points, faces) in enumerate([*visual_surfaces(pieces), posts()]):
        UsdShade.MaterialBindingAPI.Apply(_mesh(stage, f"{prim_path}/Visual/surface_{k}", points, faces)).Bind(look)

    UsdGeom.Xform.Define(stage, f"{prim_path}/Collider")
    material = sim_utils.RigidBodyMaterialCfg(static_friction=cfg.friction, dynamic_friction=cfg.friction,
                                              restitution=0.0)
    material.func(f"{prim_path}/Collider/physicsMaterial", material)
    pmat = UsdShade.Material(stage.GetPrimAtPath(f"{prim_path}/Collider/physicsMaterial"))
    for k, pc in enumerate(pieces):
        path = f"{prim_path}/Collider/piece_{k:04d}"
        prim = _mesh(stage, path, pc["points"], np.array([f for f, _ in _hull_faces(pc["points"])], dtype=np.int64))
        UsdGeom.Imageable(prim).MakeInvisible()
        schemas.define_collision_properties(path, sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0))
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(pmat, UsdShade.Tokens.strongerThanDescendants, "physics")

    schemas.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True, disable_gravity=False, linear_damping=0.5, angular_damping=1.0,
        max_depenetration_velocity=1.0))
    schemas.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=cfg.mass))
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TubeRackCfg(SpawnerCfg):
    """The generated tube rack: its colour, the friction of its collider and its mass."""

    func: Callable = spawn_tube_rack
    color: tuple[float, float, float] = (0.05, 0.10, 0.28)
    roughness: float = 0.55
    friction: float = 1.0
    mass: float = 50.0
