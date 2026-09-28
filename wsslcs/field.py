"""
Surface vector fields (WSS) on a SurfaceMesh.

The vertex vectors are projected onto the tangent planes of the vertices and
then transported into every incident triangle with the rotation used by the
original code (angle of the vector in the vertex tangent frame minus the polar
angle of the reference edge of the corner).  Inside a triangle the field is the
linear (barycentric) interpolation of the three transported corner vectors,
expressed in the local 2-D frame of the triangle.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import io_vtk
from .mesh import SurfaceMesh, TWO_PI


class SurfaceVectorField:
    def __init__(self, mesh: SurfaceMesh, vectors: np.ndarray, scale: float = 1.0, name: str = "wss"):
        self.mesh = mesh
        self.name = name
        self.scale = float(scale)
        g = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        if len(g) != mesh.n_points:
            raise ValueError(f"vector array has {len(g)} entries, mesh has {mesh.n_points} points")
        self.raw = g * self.scale                      # scaled global vectors at the vertices
        N = mesh.vertex_normals
        self.t_vec = self.raw - np.einsum("ij,ij->i", self.raw, N)[:, None] * N
        self.mag = np.linalg.norm(self.t_vec, axis=1)
        a = np.einsum("ij,ij->i", self.t_vec, mesh.vertex_T)
        b = np.einsum("ij,ij->i", self.t_vec, mesh.vertex_B)
        self.t_angle = np.mod(np.arctan2(b, a), TWO_PI)
        self.corner_vectors = self._transport()
        self._jac: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _transport(self) -> np.ndarray:
        """(M,3,2) transported vertex vectors in the local frame of each triangle."""
        mesh = self.mesh
        tri = mesh.triangles
        phi = self.t_angle[tri] - mesh.corner_begin
        phi = (phi + np.pi) % TWO_PI - np.pi
        c, s = np.cos(phi), np.sin(phi)
        e = mesh.corner_e0_local
        dx = c * e[..., 0] - s * e[..., 1]
        dy = s * e[..., 0] + c * e[..., 1]
        return np.stack([dx, dy], axis=-1) * self.mag[tri][..., None]

    def evaluate(self, tri_ids: np.ndarray, alpha: np.ndarray, D: Optional[np.ndarray] = None) -> np.ndarray:
        """Field (n,2) in the local frames of the triangles at barycentric points."""
        D = self.corner_vectors if D is None else D
        return np.einsum("nj,njk->nk", alpha, D[np.asarray(tri_ids)])

    def evaluate_global(self, tri_ids: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return self.mesh.vector_to_global(tri_ids, self.evaluate(tri_ids, alpha))

    # ------------------------------------------------------------------
    def jacobians(self) -> np.ndarray:
        """(M,2,2) Jacobian of the linear field inside every triangle (local frame)."""
        if self._jac is None:
            mesh, D = self.mesh, self.corner_vectors
            dD1 = D[:, 1] - D[:, 0]                      # (M,2) change along e1 = (x1, 0)
            dD2 = D[:, 2] - D[:, 0]                      # change along e2 = (x2, y2)
            x1, x2, y2 = mesh.x1, mesh.x2, mesh.y2
            det = x1 * y2
            # A = [dD1 dD2] * inv([[x1, x2],[0, y2]]) ; inv = 1/det [[y2, -x2],[0, x1]]
            J = np.zeros((mesh.n_tris, 2, 2))
            J[:, :, 0] = dD1 * (y2 / det)[:, None]
            J[:, :, 1] = (dD1 * (-x2 / det)[:, None] + dD2 * (x1 / det)[:, None])
            self._jac = J
        return self._jac

    def divergence(self) -> np.ndarray:
        J = self.jacobians()
        return J[:, 0, 0] + J[:, 1, 1]

    def point_divergence(self) -> np.ndarray:
        """Area weighted average of the triangle divergence at the vertices."""
        mesh = self.mesh
        div = self.divergence()
        acc = np.zeros(mesh.n_points)
        np.add.at(acc, mesh.triangles.ravel(), np.repeat(div * mesh.area, 3))
        wsum = np.zeros(mesh.n_points)
        np.add.at(wsum, mesh.triangles.ravel(), np.repeat(mesh.area, 3))
        return np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)

    def corner_vectors_global(self) -> np.ndarray:
        """(M,3,3) transported corner vectors as 3-D vectors."""
        mesh, D = self.mesh, self.corner_vectors
        return D[..., 0:1] * mesh.LX[:, None, :] + D[..., 1:2] * mesh.LY[:, None, :]

    def triangle_vectors(self) -> np.ndarray:
        """(M,3) field at the triangle centroids (3-D)."""
        return self.corner_vectors_global().mean(axis=1)

    @staticmethod
    def blend(D1: np.ndarray, D2: np.ndarray, s: float) -> np.ndarray:
        return D1 if s == 0.0 else D1 + (D2 - D1) * s

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str, vector_array: str = "wss", scale: float = 1.0,
                  mesh: Optional[SurfaceMesh] = None, verbose: bool = False):
        pts, tris, vec, used, _ = io_vtk.read_surface(path, vector_array)
        if mesh is None:
            mesh = SurfaceMesh(pts, tris, verbose=verbose)
        elif mesh.n_points != len(pts):
            raise ValueError(f"{path}: {len(pts)} points but the mesh has {mesh.n_points}")
        return cls(mesh, vec, scale=scale, name=used or vector_array)


def time_average_vectors(vector_list: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([np.asarray(v, dtype=np.float64) for v in vector_list]), axis=0)


class FieldSequence:
    """Periodic sequence of WSS files (unsteady data) with linear interpolation
    in time.  ``time`` is measured from the release instant (file
    ``index_first``) along the integration direction, exactly as in the
    original program: the frame index advances by ``index_delta`` (backward:
    ``-index_delta``) every ``delta_t_file`` and wraps around between
    ``FILE_index_start`` and ``FILE_index_end``."""

    def __init__(self, params, mesh: Optional[SurfaceMesh] = None, max_cache: int = 12):
        self.p = params
        self.steady = bool(params.steady)
        self.delta_t = float(params.delta_t_file) if not self.steady else float("inf")
        step = abs(int(params.index_delta)) or 1
        self.sign = -1 if params.backward else 1
        self.step = step
        self.start, self.end = int(params.FILE_index_start), int(params.FILE_index_end)
        # frames of one period: start, start+step, ..., <= end; the data is assumed
        # periodic, i.e. the frame after the last one is the first one again
        self.n_frames = 1 if self.steady else (self.end - self.start) // step + 1
        self.period_index = self.n_frames * step
        self.period = float("inf") if self.steady else self.n_frames * self.delta_t
        self.index_first = int(params.index_first)
        if not self.steady and (self.index_first - self.start) % step != 0:
            raise ValueError(f"index_first = {self.index_first} is not on the file index grid "
                             f"{self.start}, {self.start + step}, ...")
        self._cache: Dict[int, SurfaceVectorField] = {}
        self._order: List[int] = []
        self.max_cache = max_cache
        self.mesh = mesh
        self.vector_name = params.vector_array
        if self.mesh is None:
            first = self.field_at_index(self.index_first)
            self.mesh = first.mesh

    def wrap(self, idx: int) -> int:
        """Periodic continuation: any index (also beyond the last file or before
        the first one) is mapped into the available range."""
        if self.steady:
            return self.index_first
        return self.start + (idx - self.start) % self.period_index

    def frame_list(self):
        """The file indices of one period."""
        return [self.index_first] if self.steady else [self.start + k * self.step for k in range(self.n_frames)]

    def describe(self, integration_time=None) -> str:
        if self.steady:
            return "steady data (single file)"
        txt = f"unsteady data: {self.n_frames} files (indices {self.start} .. {self.start + (self.n_frames - 1) * self.step}), period {self.period:g}"
        if integration_time is not None:
            txt += f"; integration time {integration_time:g} = {integration_time / self.period:.2f} periods (periodic continuation)"
        return txt

    def file_name(self, idx: int) -> str:
        return self.p.data_file(None if self.steady else idx)

    def field_at_index(self, idx: int) -> SurfaceVectorField:
        idx = self.wrap(idx)
        f = self._cache.get(idx)
        if f is None:
            path = self.file_name(idx)
            if self.p.verbose:
                print(f"  loading {path}")
            f = SurfaceVectorField.from_file(path, self.vector_name, scale=self.p.WSS_SCALE, mesh=self.mesh)
            if self.mesh is None:
                self.mesh = f.mesh
            self._cache[idx] = f
            self._order.append(idx)
            while len(self._order) > self.max_cache:
                old = self._order.pop(0)
                self._cache.pop(old, None)
        return f

    def frames_at(self, time: float):
        """(field1, field2, s) with the interpolation weight s in [0,1]."""
        if self.steady:
            f = self.field_at_index(self.index_first)
            return f, f, 0.0
        k = int(math.floor(time / self.delta_t + 1e-9))
        tau = time - k * self.delta_t
        s = min(max(tau / self.delta_t, 0.0), 1.0)
        i1 = self.wrap(self.index_first + self.sign * self.step * k)
        i2 = self.wrap(i1 + self.sign * self.step)
        return self.field_at_index(i1), self.field_at_index(i2), s

    def corner_vectors(self, time: float) -> np.ndarray:
        f1, f2, s = self.frames_at(time)
        return SurfaceVectorField.blend(f1.corner_vectors, f2.corner_vectors, s)

    def raw_vectors(self, time: float) -> np.ndarray:
        f1, f2, s = self.frames_at(time)
        return SurfaceVectorField.blend(f1.raw, f2.raw, s)

    def field_at_time(self, time: float) -> SurfaceVectorField:
        """Field rebuilt from the time-interpolated vertex vectors (as the
        original ``Reinitialize`` step)."""
        f1, f2, s = self.frames_at(time)
        if s == 0.0:
            return f1
        return SurfaceVectorField(self.mesh, SurfaceVectorField.blend(f1.raw, f2.raw, s), scale=1.0,
                                  name=f1.name)

    def point_scalars(self, prefix: str, name: str, time: float) -> np.ndarray:
        """Time interpolation of a point scalar stored in companion files
        ``prefix<index><suffix>`` (used for the WSS divergence of flag_code 7)."""
        if self.steady:
            i1 = i2 = self.index_first
            s = 0.0
        else:
            k = int(math.floor(time / self.delta_t + 1e-9))
            s = min(max((time - k * self.delta_t) / self.delta_t, 0.0), 1.0)
            i1 = self.wrap(self.index_first + self.sign * self.step * k)
            i2 = self.wrap(i1 + self.sign * self.step)
        v1 = self._scalar_file(prefix, name, i1)
        v2 = self._scalar_file(prefix, name, i2) if s > 0 else v1
        return v1 + (v2 - v1) * s

    def _scalar_file(self, prefix: str, name: str, idx: int) -> np.ndarray:
        key = ("scalar", prefix, name, idx)
        val = getattr(self, "_scalar_cache", None)
        if val is None:
            self._scalar_cache = {}
        if key not in self._scalar_cache:
            path = prefix if self.steady and os.path.exists(prefix) else f"{prefix}{idx}{self.p.infile_suffix}"
            data = io_vtk.read_dataset(path)
            arr = data.GetPointData().GetArray(name)
            if arr is None:
                raise ValueError(f"{path} has no point array '{name}'")
            from vtkmodules.util import numpy_support as vnp
            self._scalar_cache[key] = vnp.vtk_to_numpy(arr).astype(np.float64).reshape(-1)
        return self._scalar_cache[key]

    def time_average(self) -> SurfaceVectorField:
        """Time averaged (TAWSS) field over one period of the data."""
        if self.steady:
            return self.field_at_index(self.index_first)
        idxs = self.frame_list()
        acc = np.zeros((self.mesh.n_points, 3))
        for i in idxs:
            acc += self.field_at_index(i).raw
        acc /= len(idxs)
        return SurfaceVectorField(self.mesh, acc, scale=1.0, name="TAWSS")
