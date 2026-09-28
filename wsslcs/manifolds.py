"""
Stable and unstable manifolds (separatrices) of the saddle-type fixed points
of a surface vector field = the WSS Lagrangian coherent structures of the
WSSET paper.

Every saddle is perturbed along +/- the eigenvector of the positive eigenvalue
and the two resulting trajectories are integrated forward in time: they trace
the unstable manifold (attracting WSS LCS).  Perturbation along +/- the
eigenvector of the negative eigenvalue and backward integration gives the
stable manifold (repelling WSS LCS).  The trajectories are integrated in arc
length (unit direction field, RK4) and are stopped when they reach another
fixed point, leave the domain, or exceed a maximum length / number of steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import io_vtk
from .field import SurfaceVectorField
from .fixedpoints import FixedPoint, SADDLE
from .mesh import SurfaceMesh
from .tracer import ACTIVE, LEFT, STUCK, Particles, advect_step

UNSTABLE, STABLE = 0, 1
KIND_NAMES = {UNSTABLE: "unstable", STABLE: "stable"}


@dataclass
class ManifoldBranch:
    saddle: int                      # id of the fixed point
    kind: int                        # UNSTABLE (forward) or STABLE (backward)
    sign: int                        # +1 / -1 perturbation direction
    points: np.ndarray               # (n,3)
    arclength: np.ndarray            # (n,)
    triangles: np.ndarray            # (n,)
    speed: np.ndarray                # (n,) field magnitude along the branch
    end_reason: str = ""
    end_fixed_point: int = -1

    @property
    def kind_name(self) -> str:
        return KIND_NAMES[self.kind]

    @property
    def length(self) -> float:
        return float(self.arclength[-1]) if len(self.arclength) else 0.0


def _seed(mesh: SurfaceMesh, fp: FixedPoint, direction2d: np.ndarray, eps: float):
    """Start point of a separatrix: the saddle moved by eps along direction2d
    (halved until the point lies in the saddle's triangle, else located on the
    surface)."""
    t = fp.triangle
    for _ in range(8):
        loc = fp.local + eps * direction2d
        alpha = mesh.bary(np.array([t]), np.array([loc[0]]), np.array([loc[1]]))[0]
        if np.all(alpha >= 0.0):
            return t, loc[0], loc[1]
        eps *= 0.5
    g = mesh.local_to_global(np.array([t]), np.array([loc[0]]), np.array([loc[1]]))
    tri, a, b, _, _ = mesh.locate(g)
    return int(tri[0]), float(a[0]), float(b[0])


def compute_manifolds(mesh: SurfaceMesh, field: SurfaceVectorField, fixed_points: List[FixedPoint],
                      step_fraction: float = 0.2, max_steps: int = 20000, max_length: Optional[float] = None,
                      perturbation: float = 0.1, capture_radius: float = 0.5,
                      max_crossings: int = 50, verbose: bool = False) -> List[ManifoldBranch]:
    saddles = [fp for fp in fixed_points if fp.type_code == SADDLE]
    if not saddles:
        return []
    if not max_length:
        max_length = 25.0 * mesh.radius
    sq_area = np.sqrt(mesh.area)
    # ---- seeds: 4 branches per saddle ------------------------------------
    tri, a, b, kinds, signs, owners, dirs = [], [], [], [], [], [], []
    for fp in saddles:
        eps = perturbation * sq_area[fp.triangle]
        for kind, evec, direction in ((UNSTABLE, fp.eigvec_out_local, 1.0), (STABLE, fp.eigvec_in_local, -1.0)):
            for sign in (1, -1):
                t, aa, bb = _seed(mesh, fp, sign * evec, eps)
                tri.append(t)
                a.append(aa)
                b.append(bb)
                kinds.append(kind)
                signs.append(sign)
                owners.append(fp.id)
                dirs.append(direction)
    parts = Particles(np.array(tri), np.array(a), np.array(b))
    n = parts.n
    dirs = np.array(dirs)
    owners = np.array(owners)
    D = field.corner_vectors
    D_at = lambda f: D

    fp_pos = np.array([fp.position for fp in fixed_points]).reshape(-1, 3)
    fp_rad = capture_radius * sq_area[[fp.triangle for fp in fixed_points]]
    fp_ids = np.array([fp.id for fp in fixed_points])

    pts = [[] for _ in range(n)]
    arcl = [[] for _ in range(n)]
    tris = [[] for _ in range(n)]
    spd = [[] for _ in range(n)]
    end_reason = [""] * n
    end_fp = np.full(n, -1, dtype=np.int64)
    pos = parts.positions(mesh)
    for i in range(n):
        pts[i].append(pos[i].copy())
        arcl[i].append(0.0)
        tris[i].append(int(parts.tri[i]))
    length = np.zeros(n)
    left_start = np.zeros(n, dtype=bool)        # moved away from the own saddle
    for it in range(max_steps):
        act = parts.status == ACTIVE
        if not np.any(act):
            break
        h = step_fraction * sq_area[parts.tri]
        res = advect_step(mesh, D_at, parts, h, dirs, integrator="rk4", normalize=True,
                          max_rounds=max_crossings)
        pos = parts.positions(mesh)
        used = res["used"]
        speed = res["speed"]
        wact = np.nonzero(act)[0]
        length[wact] += used[wact]
        # distances to the fixed points
        d = np.linalg.norm(pos[wact][:, None, :] - fp_pos[None, :, :], axis=2)   # (k, nfp)
        own = fp_ids[None, :] == owners[wact][:, None]
        far_from_own = np.all(np.where(own, d, np.inf) > 3.0 * fp_rad[None, :], axis=1)
        left_start[wact] |= far_from_own
        d_eff = np.where(own & ~left_start[wact][:, None], np.inf, d)
        within = d_eff <= fp_rad[None, :]
        for k, i in enumerate(wact):
            pts[i].append(pos[i].copy())
            arcl[i].append(length[i])
            tris[i].append(int(parts.tri[i]))
            spd[i].append(float(speed[i]))
            st = parts.status[i]
            if st == LEFT:
                end_reason[i] = "left the domain"
            elif st == STUCK:
                end_reason[i] = "zero field / vertex"
            elif np.any(within[k]):
                j = int(np.argmin(np.where(within[k], d[k], np.inf)))
                end_reason[i] = f"reached fixed point {fixed_points[j].id} ({fixed_points[j].type_name})"
                end_fp[i] = fixed_points[j].id
                parts.status[i] = STUCK
            elif length[i] >= max_length:
                end_reason[i] = "maximum length"
                parts.status[i] = STUCK
    for i in range(n):
        if parts.status[i] == ACTIVE:
            end_reason[i] = "maximum number of steps"
    branches: List[ManifoldBranch] = []
    for i in range(n):
        p = np.asarray(pts[i]).reshape(-1, 3)
        s = np.asarray(arcl[i])
        sp = np.asarray(spd[i]) if spd[i] else np.zeros(0)
        if len(sp) < len(p):
            sp = np.concatenate([np.full(len(p) - len(sp), sp[0] if len(sp) else 0.0), sp])
        branches.append(ManifoldBranch(saddle=int(owners[i]), kind=int(kinds[i]), sign=int(signs[i]),
                                       points=p, arclength=s, triangles=np.asarray(tris[i]), speed=sp,
                                       end_reason=end_reason[i], end_fixed_point=int(end_fp[i])))
    if verbose:
        for br in branches:
            print(f"  saddle {br.saddle:3d} {br.kind_name:8s} {br.sign:+d}: {len(br.points):6d} points, "
                  f"length {br.length:.4g}, {br.end_reason}")
    return branches


def manifolds_polydata(branches: List[ManifoldBranch], scale: float = 1.0):
    lines = [br.points for br in branches]
    point_arrays = {
        "arclength": [br.arclength for br in branches],
        "speed": [br.speed / scale for br in branches],
        "triangle": [br.triangles.astype(np.int32) for br in branches],
        "manifold_point": [np.full(len(br.points), br.kind, dtype=np.int32) for br in branches],
    }
    cell_arrays = {
        "manifold": np.array([br.kind for br in branches], dtype=np.int32),
        "manifold_name": [br.kind_name for br in branches],
        "saddle": np.array([br.saddle for br in branches], dtype=np.int32),
        "branch": np.array([br.sign for br in branches], dtype=np.int32),
        "length": np.array([br.length for br in branches]),
        "end_fixed_point": np.array([br.end_fixed_point for br in branches], dtype=np.int32),
        "end_reason": [br.end_reason for br in branches],
    }
    return io_vtk.polylines_polydata(lines, point_arrays, cell_arrays)
