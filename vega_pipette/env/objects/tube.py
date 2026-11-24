"""The bevelled sample tube of the pick-and-place task: Isaac Lab's cylinder with its geometry rewritten into a solid of
revolution whose bottom edge is bevelled (progress/rack.py TUBE_BEVEL), so a tube lowered slightly beside a pocket
slides into it along the bevel instead of landing on the rack.

The solid has the cylinder's radius and length and TUBE_SIDES sides: a flat bottom disc of radius r - width, a cone up
to the full radius `height` above the bottom, the wall and a flat top. Collision and visual use the same mesh (a convex
hull collider); everything else the cylinder's spawner authored (materials, collision properties, the body) is kept.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ...progress import rack

if TYPE_CHECKING:
    from pxr import Usd


def bevelled_tube_mesh(radius: float, height: float, width: float, rise: float, n: int = rack.TUBE_SIDES) -> dict:
    """The tube as a closed triangle mesh centred on its root (z from -height / 2 to +height / 2): "points" (3n + 2, 3),
    "faces" (6n, 3) counter-clockwise from outside, "normals" (6n, 3, 3) per face corner (caps flat, cone and wall
    smooth around the axis), "extent"."""
    cs = np.array([[math.cos(2.0 * math.pi * k / n), math.sin(2.0 * math.pi * k / n)] for k in range(n)])
    cs[np.abs(cs) < 1e-12] = 0.0
    z0, z1, z2 = -0.5 * height, -0.5 * height + rise, 0.5 * height

    def circle(r: float, z: float) -> np.ndarray:
        return np.concatenate([r * cs, np.full((n, 1), z)], axis=1)
    points = np.concatenate([circle(radius - width, z0), circle(radius, z1), circle(radius, z2),
                             [[0.0, 0.0, z0]], [[0.0, 0.0, z2]]])
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    wr = width / rise

    def flat(k: int) -> np.ndarray:
        return np.array([cs[k][0], cs[k][1], 0.0])

    def cone(k: int) -> np.ndarray:
        return np.array([cs[k][0], cs[k][1], -wr]) / math.sqrt(1.0 + wr * wr)
    faces, normals = [], []
    for i in range(n):
        j = (i + 1) % n
        faces.append((3 * n, j, i))
        normals.append((down, down, down))
        faces += [(i, j, n + j), (i, n + j, n + i)]
        normals += [(cone(i), cone(j), cone(j)), (cone(i), cone(j), cone(i))]
        faces += [(n + i, n + j, 2 * n + j), (n + i, 2 * n + j, 2 * n + i)]
        normals += [(flat(i), flat(j), flat(j)), (flat(i), flat(j), flat(i))]
        faces.append((3 * n + 1, 2 * n + i, 2 * n + j))
        normals.append((up, up, up))
    return {"points": points, "faces": np.array(faces, dtype=np.int64), "normals": np.array(normals, dtype=np.float64),
            "extent": ((-radius, -radius, z0), (radius, radius, z2))}


def bevel_tube(prim: Usd.Prim) -> None:
    """Rewrite the spawned tube's cylinder (<tube>/geometry/mesh) into the bevelled mesh, in place."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    geom = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath("geometry/mesh"))
    if not geom.IsValid() or geom.GetTypeName() != "Cylinder" or geom.GetAttribute("axis").Get() != "Z":
        raise RuntimeError(f"{prim.GetPath()}/geometry/mesh is not the tube's z cylinder")
    m = bevelled_tube_mesh(float(geom.GetAttribute("radius").Get()), float(geom.GetAttribute("height").Get()),
                           *rack.TUBE_BEVEL)
    for name in ("radius", "height", "axis", "extent"):
        geom.RemoveProperty(name)
    geom.SetTypeName("Mesh")
    mesh = UsdGeom.Mesh(geom)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(x) for x in p)) for p in m["points"]]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m["faces"])))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in m["faces"].reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(x) for x in v)) for v in m["normals"].reshape(-1, 3)]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*m["extent"][0]), Gf.Vec3f(*m["extent"][1])]))
    UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
