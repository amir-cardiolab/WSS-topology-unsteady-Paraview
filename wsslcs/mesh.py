"""
Triangulated surface mesh with the connectivity, normals, per-triangle local
frames and per-vertex "fans" (ordered incident triangles with the scaled polar
angles of the original code) that are needed for surface vector-field analysis.

Conventions (identical to the C++ code):

* every triangle t has a local 2-D frame with origin at its first vertex,
  ``LX`` along the edge v0->v1 and ``LY = normal x LX``; the vertices have the
  local coordinates (0,0), (x1,0) and (x2,y2);
* every vertex has a tangent frame (T, B, N); the incident triangles are
  ordered around the vertex and the corner angles are scaled by
  ``r = 2*pi / (sum of the corner angles)`` so that the fan of an interior
  vertex fills a full turn (discrete polar map).  Boundary vertices are not
  scaled (``r = 1``).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
try:
    import vtk
except ImportError:                                  # ParaView's Python: no top-level 'vtk' module
    import vtkmodules.all as vtk

from . import io_vtk

TWO_PI = 2.0 * np.pi


def unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


class SurfaceMesh:
    """Triangle mesh (points (N,3), triangles (M,3))."""

    def __init__(self, points: np.ndarray, triangles: np.ndarray, verbose: bool = False):
        self.verbose = verbose
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
                & (tris[:, 0] != tris[:, 2]) & (area2 > 0.0))
        if not np.all(good):
            warnings.warn(f"{np.count_nonzero(~good)} degenerate triangles were removed")
        self.triangle_ids_original = np.nonzero(good)[0]
        self.points = pts
        self.triangles = np.ascontiguousarray(tris[good])
        self.n_points = int(len(pts))
        self.n_tris = int(len(self.triangles))
        if self.n_tris == 0:
            raise ValueError("the mesh has no valid triangles")
        self._build_edges()
        self._build_geometry()
        self._build_frames()
        self._build_fans()
        self._locator = None
        self._locator_pd = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_edges(self) -> None:
        tri = self.triangles
        M = self.n_tris
        # edge j of a triangle joins its local vertices j and j+1
        e = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1).reshape(-1, 2)
        es = np.sort(e, axis=1)
        edges, inverse = np.unique(es, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        self.edges = edges
        self.n_edges = int(len(edges))
        self.tri_edges = inverse.reshape(M, 3)
        eid = self.tri_edges.ravel()
        tid = np.repeat(np.arange(M), 3)
        order = np.argsort(eid, kind="stable")
        eid_s, tid_s = eid[order], tid[order]
        counts = np.bincount(eid_s, minlength=self.n_edges)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        edge_tris = -np.ones((self.n_edges, 2), dtype=np.int64)
        has1 = counts >= 1
        edge_tris[has1, 0] = tid_s[starts[has1]]
        has2 = counts >= 2
        edge_tris[has2, 1] = tid_s[starts[has2] + 1]
        if np.any(counts > 2):
            warnings.warn(f"{np.count_nonzero(counts > 2)} non-manifold edges (shared by more than "
                          "two triangles); only the first two triangles are connected")
        self.edge_tris = edge_tris
        self.boundary_edge = counts == 1
        et = edge_tris[self.tri_edges]                      # (M,3,2)
        own = np.arange(M)[:, None]
        self.tri_neighbors = np.where(et[:, :, 0] == own, et[:, :, 1], et[:, :, 0])
        self.boundary_triangle = np.any(self.tri_neighbors < 0, axis=1)
        self.boundary_vertex = np.zeros(self.n_points, dtype=bool)
        self.boundary_vertex[edges[self.boundary_edge].ravel()] = True
        self.edge_length = np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)
        self.mean_edge_length = float(self.edge_length.mean())
        self.shortest_edge_length = float(self.edge_length.min())

    def _build_geometry(self) -> None:
        P, tri = self.points, self.triangles
        p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        e1, e2 = p1 - p0, p2 - p0
        cr = np.cross(e1, e2)
        area2 = np.linalg.norm(cr, axis=1)
        self.area = 0.5 * area2
        self.total_area = float(self.area.sum())
        self.mean_area = float(self.area.mean())
        normals = cr / area2[:, None]
        mn, mx = P.min(axis=0), P.max(axis=0)
        self.center = 0.5 * (mn + mx)
        self.radius = float(np.linalg.norm(self.center - mn))
        self.bounds = np.stack([mn, mx])
        # orientation convention of the original code: normals point away from
        # the centre of the bounding box (flip all if not)
        sv = float(np.sum(np.einsum("ij,ij->i", self.center - p0, normals) * self.area))
        self.normals_flipped = sv > 0
        if self.normals_flipped:
            normals = -normals
        self.tri_normals = normals
        vn = np.zeros((self.n_points, 3))
        np.add.at(vn, tri.ravel(), np.repeat(normals * self.area[:, None], 3, axis=0))
        self.vertex_normals = unit_rows(vn)
        self.vertex_area = np.zeros(self.n_points)
        np.add.at(self.vertex_area, tri.ravel(), np.repeat(self.area / 3.0, 3))

    def _build_frames(self) -> None:
        P, tri = self.points, self.triangles
        p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        e1, e2 = p1 - p0, p2 - p0
        self.LX = unit_rows(e1)
        self.LY = unit_rows(np.cross(self.tri_normals, self.LX))
        self.x1 = np.linalg.norm(e1, axis=1)
        self.x2 = np.einsum("ij,ij->i", e2, self.LX)
        self.y2 = np.einsum("ij,ij->i", e2, self.LY)
        self.local_xy = np.zeros((self.n_tris, 3, 2))
        self.local_xy[:, 1, 0] = self.x1
        self.local_xy[:, 2, 0] = self.x2
        self.local_xy[:, 2, 1] = self.y2
        self.clockwise = self.y2 < 0

    def _build_fans(self) -> None:
        N, M = self.n_points, self.n_tris
        tri, P, nb = self.triangles, self.points, self.tri_neighbors
        order = np.argsort(tri.ravel(), kind="stable")
        self.vtx_tri = (order // 3).astype(np.int64)
        self.vtx_corner = (order % 3).astype(np.int64)
        counts = np.bincount(tri.ravel(), minlength=N)
        self.vtx_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        # geometric corner angles
        ang = np.zeros((M, 3))
        for j in range(3):
            a = P[tri[:, (j + 1) % 3]] - P[tri[:, j]]
            b = P[tri[:, (j + 2) % 3]] - P[tri[:, j]]
            den = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
            ang[:, j] = np.arccos(np.clip(np.einsum("ij,ij->i", a, b) / den, -1.0, 1.0))
        self.corner_angle = ang
        VN = self.vertex_normals
        corner_begin = np.zeros((M, 3))
        corner_other = ((np.arange(3)[None, :] + 2) % 3).repeat(M, axis=0)   # default: previous vertex
        corner_in_fan = np.zeros((M, 3), dtype=bool)
        vertex_T = np.zeros((N, 3))
        vertex_B = np.zeros((N, 3))
        vertex_r = np.ones(N)
        vertex_orient = np.ones(N)
        vertex_total_angle = np.zeros(N)
        vertex_closed = np.zeros(N, dtype=bool)
        n_nonmanifold = 0
        for v in range(N):
            s0, s1 = self.vtx_start[v], self.vtx_start[v + 1]
            K = s1 - s0
            if K == 0:
                # isolated vertex: arbitrary frame
                n = VN[v] if np.linalg.norm(VN[v]) > 0 else np.array([0.0, 0.0, 1.0])
                t = np.cross(n, [1.0, 0.0, 0.0])
                if np.linalg.norm(t) < 1e-8:
                    t = np.cross(n, [0.0, 1.0, 0.0])
                vertex_T[v] = t / np.linalg.norm(t)
                vertex_B[v] = np.cross(n, vertex_T[v])
                continue
            inc_t = self.vtx_tri[s0:s1]
            inc_c = self.vtx_corner[s0:s1]
            # choose the start corner: a boundary edge if there is one
            start, e_in = -1, -1
            for k in range(K):
                t, j = inc_t[k], inc_c[k]
                if nb[t, (j + 2) % 3] < 0:
                    start, e_in = k, (j + 2) % 3
                    break
            if start < 0:
                for k in range(K):
                    t, j = inc_t[k], inc_c[k]
                    if nb[t, j] < 0:
                        start, e_in = k, j
                        break
            if start < 0:
                start, e_in = 0, (inc_c[0] + 2) % 3
            # walk around the vertex (independent of the triangle winding)
            t, j = int(inc_t[start]), int(inc_c[start])
            fan_t, fan_j, fan_other = [], [], []
            visited = set()
            closed = False
            while True:
                if t in visited:
                    break
                visited.add(t)
                other_in = (j + 1) % 3 if e_in == j else (j + 2) % 3
                fan_t.append(t)
                fan_j.append(j)
                fan_other.append(other_in)
                e_out = j if e_in != j else (j + 2) % 3
                tn = int(nb[t, e_out])
                if tn < 0:
                    break
                if tn == fan_t[0]:
                    closed = True
                    break
                w = tri[t, (j + 1) % 3] if e_out == j else tri[t, (j + 2) % 3]
                row = tri[tn]
                jn = int(np.nonzero(row == v)[0][0])
                wn = int(np.nonzero(row == w)[0][0])
                e_in = jn if wn == (jn + 1) % 3 else (jn + 2) % 3
                t, j = tn, jn
            if len(fan_t) < K:
                n_nonmanifold += 1
            fan_t = np.asarray(fan_t)
            fan_j = np.asarray(fan_j)
            fan_other = np.asarray(fan_other)
            angles = ang[fan_t, fan_j]
            total = float(angles.sum())
            r = TWO_PI / total if (closed and total > 0) else 1.0
            n = VN[v]
            # tangent frame from the reference edge of the first corner
            T = None
            for k in range(len(fan_t)):
                d0 = P[tri[fan_t[k], fan_other[k]]] - P[v]
                tt = d0 - np.dot(d0, n) * n
                nt = np.linalg.norm(tt)
                if nt > 1e-12 * max(np.linalg.norm(d0), 1e-300):
                    T = tt / nt
                    ref_k = k
                    break
            if T is None:
                T = np.cross(n, [1.0, 0.0, 0.0])
                if np.linalg.norm(T) < 1e-8:
                    T = np.cross(n, [0.0, 1.0, 0.0])
                T = T / np.linalg.norm(T)
                ref_k = 0
            B = np.cross(n, T)
            nB = np.linalg.norm(B)
            B = B / nB if nB > 0 else B
            # orientation of the fan from the first corner (exit edge vs. entry edge)
            t0, j0, o0 = fan_t[0], fan_j[0], fan_other[0]
            third = 3 - j0 - o0
            d1 = P[tri[t0, third]] - P[v]
            phi1 = np.arctan2(np.dot(d1, B), np.dot(d1, T))
            s = 1.0 if phi1 >= 0 else -1.0
            cum = np.concatenate([[0.0], np.cumsum(angles)]) * r
            # the reference edge (corner ref_k) has the polar angle 0 by construction
            begin = s * (cum[:-1] - cum[ref_k])
            corner_begin[fan_t, fan_j] = begin
            corner_other[fan_t, fan_j] = fan_other
            corner_in_fan[fan_t, fan_j] = True
            vertex_T[v], vertex_B[v] = T, B
            vertex_r[v], vertex_orient[v] = r, s
            vertex_total_angle[v] = total
            vertex_closed[v] = closed
            if len(fan_t) < K:
                # corners not reached by the walk (non-manifold vertex): use the
                # projected angle of their reference edge
                for k in range(K):
                    t, j = int(inc_t[k]), int(inc_c[k])
                    if corner_in_fan[t, j]:
                        continue
                    o = (j + 2) % 3
                    d = P[tri[t, o]] - P[v]
                    corner_begin[t, j] = np.arctan2(np.dot(d, B), np.dot(d, T))
                    corner_other[t, j] = o
        if n_nonmanifold:
            warnings.warn(f"{n_nonmanifold} non-manifold vertices (incomplete fans)")
        self.corner_begin = corner_begin
        self.corner_other = corner_other
        self.vertex_T, self.vertex_B = vertex_T, vertex_B
        self.vertex_r, self.vertex_orient = vertex_r, vertex_orient
        self.vertex_total_angle = vertex_total_angle
        self.vertex_closed_fan = vertex_closed
        # unit direction (local 2-D) of the reference edge of every corner
        rows = np.arange(M)[:, None]
        d = P[tri[rows, corner_other]] - P[tri][:, :, :]          # (M,3,3)
        ex = np.einsum("mjk,mk->mj", d, self.LX)
        ey = np.einsum("mjk,mk->mj", d, self.LY)
        e0 = np.stack([ex, ey], axis=-1)
        self.corner_e0_local = unit_rows(e0)

    # ------------------------------------------------------------------
    # coordinate transformations (all vectorised)
    # ------------------------------------------------------------------
    def bary(self, tri_ids: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Barycentric coordinates (n,3) of local points (a,b) in triangles tri_ids."""
        t = np.asarray(tri_ids)
        a2 = b / self.y2[t]
        a1 = (a - a2 * self.x2[t]) / self.x1[t]
        return np.stack([1.0 - a1 - a2, a1, a2], axis=-1)

    def local_from_bary(self, tri_ids: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(tri_ids)
        a = alpha[:, 1] * self.x1[t] + alpha[:, 2] * self.x2[t]
        b = alpha[:, 2] * self.y2[t]
        return a, b

    def local_to_global(self, tri_ids: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        t = np.asarray(tri_ids)
        return (self.points[self.triangles[t, 0]] + np.asarray(a)[:, None] * self.LX[t]
                + np.asarray(b)[:, None] * self.LY[t])

    def global_to_local(self, tri_ids: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(tri_ids)
        d = np.asarray(pts).reshape(-1, 3) - self.points[self.triangles[t, 0]]
        return np.einsum("ij,ij->i", d, self.LX[t]), np.einsum("ij,ij->i", d, self.LY[t])

    def vector_to_global(self, tri_ids: np.ndarray, v2: np.ndarray) -> np.ndarray:
        t = np.asarray(tri_ids)
        v2 = np.asarray(v2).reshape(-1, 2)
        return v2[:, 0:1] * self.LX[t] + v2[:, 1:2] * self.LY[t]

    def vector_to_local(self, tri_ids: np.ndarray, v3: np.ndarray) -> np.ndarray:
        t = np.asarray(tri_ids)
        v3 = np.asarray(v3).reshape(-1, 3)
        return np.stack([np.einsum("ij,ij->i", v3, self.LX[t]),
                         np.einsum("ij,ij->i", v3, self.LY[t])], axis=-1)

    def centroids(self) -> np.ndarray:
        return self.points[self.triangles].mean(axis=1)

    def cell_to_point(self, values: np.ndarray) -> np.ndarray:
        """Area weighted average of a per-triangle quantity at the vertices."""
        values = np.asarray(values, dtype=np.float64)
        acc = np.zeros(self.n_points if values.ndim == 1 else (self.n_points, values.shape[1]))
        w = np.repeat(self.area, 3)
        if values.ndim == 1:
            np.add.at(acc, self.triangles.ravel(), np.repeat(values * self.area, 3))
        else:
            np.add.at(acc, self.triangles.ravel(), np.repeat(values * self.area[:, None], 3, axis=0))
        wsum = np.zeros(self.n_points)
        np.add.at(wsum, self.triangles.ravel(), w)
        if values.ndim == 1:
            return np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
        return np.divide(acc, wsum[:, None], out=np.zeros_like(acc), where=wsum[:, None] > 0)

    def inside(self, alpha: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return np.all(alpha >= -tol, axis=-1)

    # ------------------------------------------------------------------
    # seeds / point location
    # ------------------------------------------------------------------
    def vertex_seeds(self, vertex_ids: Optional[Sequence[int]] = None):
        """A triangle and local coordinates for points located at vertices."""
        ids = np.arange(self.n_points) if vertex_ids is None else np.asarray(vertex_ids, dtype=np.int64)
        if np.any(self.vtx_start[ids + 1] == self.vtx_start[ids]):
            raise ValueError("some vertices are not used by any triangle")
        tri = self.vtx_tri[self.vtx_start[ids]]
        corner = self.vtx_corner[self.vtx_start[ids]]
        alpha = np.zeros((len(ids), 3))
        alpha[np.arange(len(ids)), corner] = 1.0
        a, b = self.local_from_bary(tri, alpha)
        return tri, a, b

    def centroid_seeds(self, tri_ids: Optional[Sequence[int]] = None):
        ids = np.arange(self.n_tris) if tri_ids is None else np.asarray(tri_ids, dtype=np.int64)
        alpha = np.full((len(ids), 3), 1.0 / 3.0)
        a, b = self.local_from_bary(ids, alpha)
        return ids, a, b

    def _get_locator(self):
        if self._locator is None:
            self._locator_pd = io_vtk.surface_polydata(self.points, self.triangles)
            loc = vtk.vtkStaticCellLocator()
            loc.SetDataSet(self._locator_pd)
            loc.BuildLocator()
            self._locator = loc
        return self._locator

    def locate(self, pts: np.ndarray):
        """Closest surface point of arbitrary 3-D points.
        Returns (tri, a, b, projected points, distances)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        loc = self._get_locator()
        n = len(pts)
        tri = np.zeros(n, dtype=np.int64)
        proj = np.zeros((n, 3))
        dist = np.zeros(n)
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        d2 = vtk.reference(0.0)
        for i in range(n):
            loc.FindClosestPoint(pts[i].tolist(), closest, cell_id, sub_id, d2)
            tri[i] = int(cell_id)
            proj[i] = closest
            dist[i] = np.sqrt(float(d2))
        a, b = self.global_to_local(tri, proj)
        alpha = np.clip(self.bary(tri, a, b), 0.0, None)
        alpha /= alpha.sum(axis=1, keepdims=True)
        a, b = self.local_from_bary(tri, alpha)
        return tri, a, b, proj, dist

    # ------------------------------------------------------------------
    def to_polydata(self, point_arrays=None, cell_arrays=None, field_arrays=None) -> vtk.vtkPolyData:
        return io_vtk.surface_polydata(self.points, self.triangles, point_arrays, cell_arrays, field_arrays)

    def summary(self) -> str:
        return (f"mesh: {self.n_points} points, {self.n_tris} triangles, {self.n_edges} edges, "
                f"{int(self.boundary_edge.sum())} boundary edges, "
                f"mean edge length {self.mean_edge_length:.4g}, radius {self.radius:.4g}")

    @classmethod
    def from_polydata(cls, pd: vtk.vtkPolyData, **kw) -> "SurfaceMesh":
        pd = io_vtk.as_triangle_polydata(pd)
        return cls(io_vtk.polydata_points(pd), io_vtk.polydata_triangles(pd), **kw)

    @classmethod
    def from_file(cls, path: str, **kw) -> "SurfaceMesh":
        pts, tris, _, _, _ = io_vtk.read_surface(path, vector_array=None, required=False)
        return cls(pts, tris, **kw)
