"""
Fixed points (critical points) of a surface vector field.

1. the Poincare index of every triangle is computed from the winding of the
   three transported corner vectors (index +1 or -1 marks a fixed point);
2. the position inside the triangle is the zero of the linear field
   (3x3 linear system for the barycentric coordinates);
3. the Jacobian of the linear field gives the eigenvalues/eigenvectors and
   the type: source, sink, saddle, center, attracting/repelling focus.

Boundary triangles and triangles with a (numerically) zero vertex vector are
skipped, as in the original code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import io_vtk
from .field import SurfaceVectorField
from .mesh import TWO_PI

# type codes of the original program
UNKNOWN, SOURCE, SINK, SADDLE, CENTER, ATTRACTING_FOCUS, REPELLING_FOCUS = 0, 1, 2, 3, 4, 6, 7
TYPE_NAMES = {UNKNOWN: "unknown", SOURCE: "source", SINK: "sink", SADDLE: "saddle", CENTER: "center",
              ATTRACTING_FOCUS: "attracting focus", REPELLING_FOCUS: "repelling focus"}


@dataclass
class FixedPoint:
    id: int
    triangle: int
    position: np.ndarray            # (3,) global coordinates
    local: np.ndarray               # (2,) coordinates in the triangle frame
    alpha: np.ndarray               # (3,) barycentric coordinates
    poincare_index: int
    type_code: int
    type_name: str
    eigenvalues: np.ndarray         # (2,) complex
    jacobian: np.ndarray            # (2,2) local frame
    eigvec_out_local: Optional[np.ndarray] = None   # saddles: eigenvector of the positive eigenvalue
    eigvec_in_local: Optional[np.ndarray] = None    # saddles: eigenvector of the negative eigenvalue
    eigvec_out: Optional[np.ndarray] = None         # the same in 3-D
    eigvec_in: Optional[np.ndarray] = None
    residual: float = 0.0           # |field| at the located point (should be ~0)
    located: bool = True

    @property
    def is_saddle(self) -> bool:
        return self.type_code == SADDLE

    @property
    def eigenvalue_total(self) -> float:
        return float(np.sum(np.abs(self.eigenvalues)))


def poincare_indices(field: SurfaceVectorField, exclude_boundary: bool = True,
                     zero_tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Per-triangle Poincare index (0, +1, -1) and the mask of the triangles
    that were examined."""
    mesh, D = field.mesh, field.corner_vectors
    ang = np.arctan2(D[..., 1], D[..., 0])                    # (M,3)
    diff = ang - np.roll(ang, 1, axis=1)                        # Polar[j] - Polar[j-1]
    diff = (diff + np.pi) % TWO_PI - np.pi
    total = diff.sum(axis=1)
    total = np.where(mesh.clockwise, -total, total)
    index = np.rint(total / TWO_PI).astype(np.int64)
    examined = np.ones(mesh.n_tris, dtype=bool)
    if exclude_boundary:
        examined &= ~mesh.boundary_triangle
    zero_vertex = np.any(field.mag[mesh.triangles] < zero_tol, axis=1)
    examined &= ~zero_vertex
    index = np.where(examined, index, 0)
    return index, examined


def classify(J: np.ndarray, rel_tol: float = 1e-8):
    """Type code, eigenvalues and eigenvectors of a 2x2 Jacobian."""
    evals, evecs = np.linalg.eig(J)
    scale = max(float(np.max(np.abs(evals))), 1e-300)
    im = np.abs(evals.imag) <= rel_tol * scale
    re = evals.real
    if np.all(im):
        r1, r2 = float(re[0]), float(re[1])
        small = rel_tol * scale
        if abs(r1) <= small or abs(r2) <= small:
            code = UNKNOWN
        elif r1 > 0 and r2 > 0:
            code = SOURCE
        elif r1 < 0 and r2 < 0:
            code = SINK
        else:
            code = SADDLE
    else:
        r = float(re[0])
        if abs(r) <= rel_tol * scale:
            code = CENTER
        elif r < 0:
            code = ATTRACTING_FOCUS
        else:
            code = REPELLING_FOCUS
    return code, evals, evecs


def find_fixed_points(field: SurfaceVectorField, exclude_boundary: bool = True,
                      zero_tol: float = 1e-10, verbose: bool = False) -> Tuple[List[FixedPoint], np.ndarray]:
    """Locate and classify all fixed points.  Returns the list and the
    per-triangle Poincare index array."""
    mesh, D = field.mesh, field.corner_vectors
    index, _ = poincare_indices(field, exclude_boundary, zero_tol)
    cand = np.nonzero(index != 0)[0]
    fps: List[FixedPoint] = []
    if cand.size == 0:
        return fps, index
    # zero of the linear field: sum_j alpha_j D_j = 0, sum_j alpha_j = 1
    A = np.zeros((cand.size, 3, 3))
    A[:, 0, :] = D[cand, :, 0]
    A[:, 1, :] = D[cand, :, 1]
    A[:, 2, :] = 1.0
    rhs = np.zeros((cand.size, 3))
    rhs[:, 2] = 1.0
    alpha = np.full((cand.size, 3), 1.0 / 3.0)
    located = np.zeros(cand.size, dtype=bool)
    try:
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        ok = np.isfinite(sol).all(axis=1) & np.all(sol >= -1e-6, axis=1) & np.all(sol <= 1.0 + 1e-6, axis=1)
        alpha[ok] = np.clip(sol[ok], 0.0, None)
        alpha[ok] /= alpha[ok].sum(axis=1, keepdims=True)
        located = ok
    except np.linalg.LinAlgError:
        for i in range(cand.size):
            try:
                sol = np.linalg.solve(A[i], rhs[i])
                if np.isfinite(sol).all() and np.all(sol >= -1e-6) and np.all(sol <= 1 + 1e-6):
                    alpha[i] = np.clip(sol, 0, None) / np.clip(sol, 0, None).sum()
                    located[i] = True
            except np.linalg.LinAlgError:
                pass
    a, b = mesh.local_from_bary(cand, alpha)
    pos = mesh.local_to_global(cand, a, b)
    J_all = field.jacobians()[cand]
    resid = np.linalg.norm(field.evaluate(cand, alpha), axis=1)
    for i, t in enumerate(cand):
        J = J_all[i]
        code, evals, evecs = classify(J)
        fp = FixedPoint(id=len(fps), triangle=int(t), position=pos[i], local=np.array([a[i], b[i]]),
                        alpha=alpha[i], poincare_index=int(index[t]), type_code=code,
                        type_name=TYPE_NAMES[code], eigenvalues=evals, jacobian=J,
                        residual=float(resid[i]), located=bool(located[i]))
        if code == SADDLE:
            order = np.argsort(evals.real)
            e_in = np.real(evecs[:, order[0]])
            e_out = np.real(evecs[:, order[1]])
            e_in /= np.linalg.norm(e_in)
            e_out /= np.linalg.norm(e_out)
            fp.eigvec_in_local, fp.eigvec_out_local = e_in, e_out
            fp.eigvec_in = mesh.vector_to_global(np.array([t]), e_in[None, :])[0]
            fp.eigvec_out = mesh.vector_to_global(np.array([t]), e_out[None, :])[0]
        fps.append(fp)
    fps = _merge_duplicates(fps, 1e-6 * mesh.mean_edge_length)
    if verbose:
        counts: Dict[str, int] = {}
        for fp in fps:
            counts[fp.type_name] = counts.get(fp.type_name, 0) + 1
        print(f"  {len(fps)} fixed points: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return fps, index


def _merge_duplicates(fps: List[FixedPoint], tol: float) -> List[FixedPoint]:
    """A zero lying exactly on a shared edge is reported by both triangles;
    keep only one of them (the one with the smaller residual)."""
    if len(fps) < 2:
        return fps
    keep: List[FixedPoint] = []
    for fp in sorted(fps, key=lambda f: f.residual):
        if all(np.linalg.norm(fp.position - k.position) > tol for k in keep):
            keep.append(fp)
    keep.sort(key=lambda f: f.triangle)
    for i, fp in enumerate(keep):
        fp.id = i
    return keep


def fixed_points_polydata(fps: List[FixedPoint], scale: float = 1.0):
    """vtkPolyData with one vertex per fixed point and all its properties."""
    n = len(fps)
    pos = np.array([fp.position for fp in fps]).reshape(-1, 3)
    arrays = {
        "type": np.array([fp.type_code for fp in fps], dtype=np.int32),
        "type_name": [fp.type_name for fp in fps],
        "poincare_index": np.array([fp.poincare_index for fp in fps], dtype=np.int32),
        "triangle": np.array([fp.triangle for fp in fps], dtype=np.int32),
        "eigenvalue_real": np.array([fp.eigenvalues.real for fp in fps]).reshape(n, 2) / scale,
        "eigenvalue_imag": np.array([fp.eigenvalues.imag for fp in fps]).reshape(n, 2) / scale,
        "eigenvalue_total": np.array([fp.eigenvalue_total for fp in fps]) / scale,
        "jacobian": np.array([fp.jacobian.ravel() for fp in fps]).reshape(n, 4) / scale,
        "eigvec_outgoing": np.array([fp.eigvec_out if fp.eigvec_out is not None else np.zeros(3) for fp in fps]).reshape(n, 3),
        "eigvec_incoming": np.array([fp.eigvec_in if fp.eigvec_in is not None else np.zeros(3) for fp in fps]).reshape(n, 3),
        "residual": np.array([fp.residual for fp in fps]) / scale,
        "located": np.array([fp.located for fp in fps], dtype=np.int32),
    }
    return io_vtk.points_polydata(pos, arrays)


def summarize(fps: List[FixedPoint]) -> str:
    lines = [f"{'id':>3} {'type':<17} {'index':>5} {'triangle':>8} {'x':>10} {'y':>10} {'z':>10} {'|lambda|':>10}"]
    for fp in fps:
        x, y, z = fp.position
        lines.append(f"{fp.id:>3} {fp.type_name:<17} {fp.poincare_index:>5} {fp.triangle:>8} "
                     f"{x:>10.4f} {y:>10.4f} {z:>10.4f} {fp.eigenvalue_total:>10.4g}")
    return "\n".join(lines)
