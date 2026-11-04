"""02B physical benchmark: the no-step (capture-point) push limit of a posture.

For a linear inverted pendulum with COM height h, an instantaneous horizontal
velocity change dv moves the capture point by dv / w0, w0 = sqrt(g / h). Without
stepping, and ignoring upper-body angular momentum (hip/head strategy), the
robot can only stop if the capture point stays inside the support polygon. The
bound along direction u is therefore dv_max(u) = w0 * d(u), where d(u) is the
distance from the COM ground projection to the polygon edge along u.

The polygon is the convex hull of sole-mesh vertices within ``band`` of the
lowest sole point; this is geometry, not a measured pressure distribution.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81


def convex_hull(points):
    """Counter-clockwise 2-D hull vertices (SciPy/Qhull; sole meshes have ~5k hull points)."""
    from scipy.spatial import ConvexHull
    pts = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(pts) < 3:
        return pts
    return pts[ConvexHull(pts).vertices]


def support_polygon(sim, data, band=0.002):
    """Convex hull (x, y) of both soles' bottom vertices in world frame."""
    m = sim.m
    pts = []
    for g in sim.feet:
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pts.append(v[v[:, 2] <= v[:, 2].min() + band, :2])
    return convex_hull(np.concatenate(pts))


def ray_exit(poly, origin, u):
    """Distance from origin (inside poly) to the boundary along unit vector u."""
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        den = u[0] * (-e[1]) + u[1] * e[0]  # det([u, -e])
        if abs(den) < 1e-12:
            continue
        w = a - origin
        t = (w[0] * (-e[1]) + w[1] * e[0]) / den
        s = (u[0] * w[1] - u[1] * w[0]) / den
        if t >= 0 and -1e-9 <= s <= 1 + 1e-9:
            best = min(best, t)
    return best


def inside(poly, p):
    n = len(poly)
    return all(np.cross(poly[(i + 1) % n] - poly[i], p - poly[i]) >= -1e-12 for i in range(n))


def capture_bounds(sim, data, directions_rad):
    """{direction: dv_max} plus the geometry used, for the posture in ``data``."""
    com = data.subtree_com[sim.trunk].copy()
    poly = support_polygon(sim, data)
    w0 = math.sqrt(G / com[2])
    c = com[:2]
    ok = inside(poly, c)
    bounds = [w0 * ray_exit(poly, c, np.array([math.cos(a), math.sin(a)])) if ok else 0.0 for a in directions_rad]
    return dict(com=com.tolist(), omega0=w0, polygon=poly.tolist(), com_inside=bool(ok), bounds=bounds,
                directions_rad=list(directions_rad))
