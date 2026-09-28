"""
Integration of surface tracers ("WSS trajectories").

The tracers live inside the triangles (triangle id + local 2-D coordinates).
One time step of a tracer is integrated inside its current triangle with the
linear field of that triangle (explicit Euler or RK4).  When the step leaves
the triangle the exact exit point on the edge is computed, the tracer is moved
into the neighbouring triangle (the point is re-expressed in the local frame
of the neighbour) and the remaining part of the time step is integrated there.
Tracers reaching a boundary edge leave the domain.  A tracer that reaches a
vertex continues in the incident triangle into which the transported vertex
vector points (as ``pass_vertex`` in the original code).

Time spent by every tracer in every triangle is accumulated exactly (this is
the exposure time of Eq. (5) of the WSSET paper).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np

from .mesh import SurfaceMesh

ACTIVE, LEFT, STUCK, ERROR, WAITING, OUT_NORMAL = 0, 1, 2, 3, 4, 5


class RunCancelled(Exception):
    """Raised when a progress callback asks to stop the computation."""

STATUS_NAMES = {ACTIVE: "active", LEFT: "left the domain", STUCK: "stuck (fixed point / vertex)",
                ERROR: "error", WAITING: "not yet released", OUT_NORMAL: "left the near-wall region"}


class Particles:
    """A set of surface tracers."""

    def __init__(self, tri: np.ndarray, a: np.ndarray, b: np.ndarray):
        self.tri = np.asarray(tri, dtype=np.int64).copy()
        self.a = np.asarray(a, dtype=np.float64).copy()
        self.b = np.asarray(b, dtype=np.float64).copy()
        n = len(self.tri)
        self.status = np.zeros(n, dtype=np.int64)
        self.prev_edge = np.full(n, -1, dtype=np.int64)
        self.slide = np.zeros(n, dtype=bool)

    @property
    def n(self) -> int:
        return len(self.tri)

    @property
    def active(self) -> np.ndarray:
        return self.status == ACTIVE

    @property
    def inside(self) -> np.ndarray:
        return (self.status == ACTIVE) | (self.status == STUCK)

    def positions(self, mesh: SurfaceMesh) -> np.ndarray:
        return mesh.local_to_global(self.tri, self.a, self.b)

    def bary(self, mesh: SurfaceMesh) -> np.ndarray:
        return mesh.bary(self.tri, self.a, self.b)

    @classmethod
    def from_vertices(cls, mesh: SurfaceMesh, ids=None) -> "Particles":
        return cls(*mesh.vertex_seeds(ids))

    @classmethod
    def from_centroids(cls, mesh: SurfaceMesh, ids=None) -> "Particles":
        return cls(*mesh.centroid_seeds(ids))

    @classmethod
    def from_points(cls, mesh: SurfaceMesh, pts: np.ndarray) -> "Particles":
        tri, a, b, _, _ = mesh.locate(pts)
        return cls(tri, a, b)

    @classmethod
    def concatenate(cls, parts) -> "Particles":
        p = cls(np.concatenate([q.tri for q in parts]), np.concatenate([q.a for q in parts]),
                np.concatenate([q.b for q in parts]))
        p.status = np.concatenate([q.status for q in parts])
        p.prev_edge = np.concatenate([q.prev_edge for q in parts])
        p.slide = np.concatenate([q.slide for q in parts])
        return p

    def copy(self) -> "Particles":
        p = Particles(self.tri, self.a, self.b)
        p.status = self.status.copy()
        p.prev_edge = self.prev_edge.copy()
        p.slide = self.slide.copy()
        return p


def _unit2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _vertex_transfer(mesh: SurfaceMesh, D: np.ndarray, v: int, direction: float):
    """Triangle (and its corner index) into which the transported vertex vector
    of vertex ``v`` points.  Returns (tri, corner) or None."""
    s0, s1 = mesh.vtx_start[v], mesh.vtx_start[v + 1]
    best, best_margin = None, -np.inf
    for t, j in zip(mesh.vtx_tri[s0:s1], mesh.vtx_corner[s0:s1]):
        d = direction * D[t, j]
        nd = np.hypot(d[0], d[1])
        if nd < 1e-300:
            continue
        d = d / nd
        xy = mesh.local_xy[t]
        e1 = xy[(j + 1) % 3] - xy[j]
        e2 = xy[(j + 2) % 3] - xy[j]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-300:
            continue
        al = (d[0] * e2[1] - d[1] * e2[0]) / det
        be = (e1[0] * d[1] - e1[1] * d[0]) / det
        margin = min(al * np.hypot(*e1), be * np.hypot(*e2))
        if margin > best_margin:
            best, best_margin = (int(t), int(j)), margin
    if best is None:
        return None, "zero"
    if best_margin < -1e-9:
        return None, "outside"
    return best, "ok"


def advect_step(mesh: SurfaceMesh,
                D_at: Callable[[float], np.ndarray],
                particles: Particles,
                h: Union[float, np.ndarray],
                direction: Union[float, np.ndarray] = 1.0,
                integrator: str = "euler",
                et_accum: Optional[np.ndarray] = None,
                speed_scale: Optional[np.ndarray] = None,
                normalize: bool = False,
                max_rounds: int = 50,
                inside_tol: float = 1e-9,
                vertex_tol: float = 1e-9,
                nudge: float = 0.0) -> Dict[str, np.ndarray]:
    """Advance all active tracers by one time step.

    Parameters
    ----------
    D_at : callable
        ``D_at(f)`` returns the (M,3,2) corner-vector field at the time
        ``t + f*h`` (``f`` in [0, 1]); for a steady field it returns the same
        array every time.
    h, direction : scalar or per-tracer arrays
        step size and integration direction (+1 forward, -1 backward)
    et_accum : (M,) array or None
        if given, the time spent in every triangle is added to it
    speed_scale : (n,) array or None
        per-tracer multiplier of the velocity (wall-normal distance models)
    normalize : bool
        integrate the unit direction field (arc-length parametrisation,
        used for the manifolds)

    Returns a dict with the time actually used by every tracer (``used``), the
    field magnitude at the start of the step (``speed``) and counters.
    """
    n = particles.n
    hh = np.broadcast_to(np.asarray(h, dtype=np.float64), (n,)).astype(np.float64)
    dd = np.broadcast_to(np.asarray(direction, dtype=np.float64), (n,)).astype(np.float64)
    used = np.zeros(n)
    speed = np.zeros(n)
    act = np.nonzero(particles.status == ACTIVE)[0]
    result = {"used": used, "speed": speed, "n_left": 0, "n_stuck": 0}
    if act.size == 0:
        return result
    D0 = D_at(0.0)
    if integrator == "rk4":
        Dh, D1 = D_at(0.5), D_at(1.0)
    else:
        Dh = D1 = D0

    tri = particles.tri[act].copy()
    a = particles.a[act].copy()
    b = particles.b[act].copy()
    rem = hh[act].copy()
    dirn = dd[act]
    status = particles.status[act].copy()
    prev_edge = particles.prev_edge[act].copy()
    slide = particles.slide[act].copy()
    ss = None if speed_scale is None else np.asarray(speed_scale)[act]
    work = np.arange(act.size)
    n_left = n_stuck = 0

    def velocity(t, aa, bb, D, w):
        al = mesh.bary(t, aa, bb)
        v = np.einsum("nj,njk->nk", al, D[t])
        if normalize:
            v = _unit2(v)
        if ss is not None:
            v = v * ss[w][:, None]
        return v

    def rk4_step(t, aa, bb, rr, dd, w, k1=None):
        if k1 is None:
            k1 = velocity(t, aa, bb, D0, w) * dd[:, None]
        k2 = velocity(t, aa + 0.5 * rr * k1[:, 0], bb + 0.5 * rr * k1[:, 1], Dh, w) * dd[:, None]
        k3 = velocity(t, aa + 0.5 * rr * k2[:, 0], bb + 0.5 * rr * k2[:, 1], Dh, w) * dd[:, None]
        k4 = velocity(t, aa + rr * k3[:, 0], bb + rr * k3[:, 1], D1, w) * dd[:, None]
        step = (rr / 6.0)[:, None] * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return aa + step[:, 0], bb + step[:, 1]

    for _round in range(max_rounds):
        if work.size == 0:
            break
        t = tri[work]
        ap, bp, r, dw = a[work], b[work], rem[work], dirn[work]
        alpha_p = mesh.bary(t, ap, bp)
        v1 = velocity(t, ap, bp, D0, work)
        if _round == 0:
            speed[act[work]] = np.linalg.norm(np.einsum("nj,njk->nk", alpha_p, D0[t]), axis=1)
        sl = slide[work]
        if np.any(sl):
            # slide along the previous edge: project the velocity on the edge
            ws = np.nonzero(sl)[0]
            ts = t[ws]
            eid = prev_edge[work[ws]]
            loc = np.argmax(mesh.tri_edges[ts] == eid[:, None], axis=1)
            xy = mesh.local_xy[ts]
            e = xy[np.arange(len(ws)), (loc + 1) % 3] - xy[np.arange(len(ws)), loc]
            e = _unit2(e)
            v1[ws] = np.einsum("ij,ij->i", v1[ws], e)[:, None] * e
        if integrator == "rk4":
            qa, qb = rk4_step(t, ap, bp, r, dw, work, k1=v1 * dw[:, None])
            if np.any(sl):
                qa[sl] = ap[sl] + r[sl] * dw[sl] * v1[sl, 0]
                qb[sl] = bp[sl] + r[sl] * dw[sl] * v1[sl, 1]
        else:
            qa, qb = ap + r * dw * v1[:, 0], bp + r * dw * v1[:, 1]
        alpha_q = mesh.bary(t, qa, qb)
        inside = np.all(alpha_q >= -inside_tol, axis=1)

        # ---- tracers that stay inside their triangle ---------------------
        if np.any(inside):
            wi = work[inside]
            a[wi], b[wi] = qa[inside], qb[inside]
            used[act[wi]] += r[inside]
            if et_accum is not None:
                np.add.at(et_accum, t[inside], r[inside])
            rem[wi] = 0.0
            slide[wi] = False
        oo = np.nonzero(~inside)[0]
        if oo.size == 0:
            work = work[:0]
            break

        # ---- tracers leaving their triangle ------------------------------
        wo = work[oo]
        t_o = t[oo]
        ap_o, aq_o = alpha_p[oo], alpha_q[oo]
        neg = aq_o < -inside_tol
        denom = ap_o - aq_o
        denom = np.where(denom == 0.0, 1.0, denom)
        s_j = np.where(neg, ap_o / denom, np.inf)
        s_j = np.clip(s_j, 0.0, np.inf)
        jstar = np.argmin(s_j, axis=1)
        rows = np.arange(oo.size)
        s = np.clip(s_j[rows, jstar], 0.0, 1.0)
        alpha_x = ap_o + s[:, None] * (aq_o - ap_o)
        if integrator == "rk4":
            # the exit fraction of the straight chord is refined on the curved
            # RK4 path (two secant iterations on alpha_j(q(s)) = 0)
            ref = np.nonzero(~sl[oo])[0]
            if ref.size:
                t_r, w_r = t_o[ref], wo[ref]
                a_r, b_r = ap[oo][ref], bp[oo][ref]
                r_r, dw_r = r[oo][ref], dw[oo][ref]
                j_r = jstar[ref]
                f_p = ap_o[ref, j_r]
                s_r = s[ref]
                nr = np.arange(ref.size)
                for _it in range(2):
                    qa1, qb1 = rk4_step(t_r, a_r, b_r, s_r * r_r, dw_r, w_r)
                    f_1 = mesh.bary(t_r, qa1, qb1)[nr, j_r]
                    den = f_p - f_1
                    good = np.abs(den) > 1e-300
                    s_r = np.clip(np.where(good, s_r * f_p / np.where(good, den, 1.0), s_r), 0.0, 1.0)
                qa1, qb1 = rk4_step(t_r, a_r, b_r, s_r * r_r, dw_r, w_r)
                alpha_x[ref] = mesh.bary(t_r, qa1, qb1)
                s[ref] = s_r
        alpha_x[rows, jstar] = 0.0
        alpha_x = np.clip(alpha_x, 0.0, None)
        alpha_x /= alpha_x.sum(axis=1, keepdims=True)
        dt_part = s * r[oo]
        used[act[wo]] += dt_part
        if et_accum is not None:
            np.add.at(et_accum, t_o, dt_part)
        rem[wo] = np.maximum(rem[wo] - dt_part, 0.0)
        edge_local = (jstar + 1) % 3
        edge_id = mesh.tri_edges[t_o, edge_local]
        nbr = mesh.tri_neighbors[t_o, edge_local]
        kmax = np.argmax(alpha_x, axis=1)
        at_vertex = alpha_x[rows, kmax] > 1.0 - vertex_tol
        same_edge = (edge_id == prev_edge[wo]) & (s < 1e-9) & ~at_vertex & ~slide[wo]

        # position on the exit edge (in the current triangle)
        ax, bx = mesh.local_from_bary(t_o, alpha_x)
        a[wo], b[wo] = ax, bx

        # (a) no progress through an edge already crossed: slide along it
        if np.any(same_edge):
            ws = wo[same_edge]
            slide[ws] = True
            prev_edge[ws] = edge_id[same_edge]
        # (b) vertex crossings (rare, handled one by one)
        vtx = np.nonzero(at_vertex & ~same_edge)[0]
        for i in vtx:
            w = wo[i]
            v = int(mesh.triangles[t_o[i], kmax[i]])
            best, why = _vertex_transfer(mesh, D0, v, dirn[w])
            if best is None:
                if why == "outside" or mesh.boundary_vertex[v]:
                    status[w] = LEFT
                    n_left += 1
                else:
                    status[w] = STUCK
                    n_stuck += 1
                rem[w] = 0.0
                continue
            tn, jn = best
            alpha_n = np.full(3, nudge)
            alpha_n[jn] = 1.0 - 2.0 * nudge
            an, bn = mesh.local_from_bary(np.array([tn]), alpha_n[None, :])
            tri[w], a[w], b[w] = tn, an[0], bn[0]
            prev_edge[w] = -1
            slide[w] = False
        # (c) ordinary edge crossings
        cross = ~at_vertex & ~same_edge
        if np.any(cross):
            wc = wo[cross]
            leave = nbr[cross] < 0
            if np.any(leave):
                wl = wc[leave]
                status[wl] = LEFT
                rem[wl] = 0.0
                n_left += int(leave.sum())
            go = ~leave
            if np.any(go):
                wg = wc[go]
                tn = nbr[cross][go]
                g = mesh.local_to_global(t_o[cross][go], ax[cross][go], bx[cross][go])
                an, bn = mesh.global_to_local(tn, g)
                alpha_n = np.clip(mesh.bary(tn, an, bn), 0.0, None)
                alpha_n /= alpha_n.sum(axis=1, keepdims=True)
                an, bn = mesh.local_from_bary(tn, alpha_n)
                tri[wg], a[wg], b[wg] = tn, an, bn
                prev_edge[wg] = edge_id[cross][go]
                slide[wg] = False
        work = wo[(rem[wo] > 1e-14 * np.maximum(hh[act[wo]], 1e-300)) & (status[wo] == ACTIVE)]

    if work.size:
        # could not finish the step (ping-pong between triangles): freeze
        status[work] = STUCK
        n_stuck += int(work.size)

    particles.tri[act] = tri
    particles.a[act] = a
    particles.b[act] = b
    particles.status[act] = status
    particles.prev_edge[act] = prev_edge
    particles.slide[act] = slide
    result["n_left"], result["n_stuck"] = n_left, n_stuck
    return result


class Trajectory:
    """Record of one tracer path."""

    def __init__(self):
        self.points = []
        self.times = []
        self.triangles = []
        self.speed = []
        self.status = ACTIVE
        self.end_reason = ""

    def as_arrays(self):
        return (np.asarray(self.points, dtype=np.float64).reshape(-1, 3), np.asarray(self.times, dtype=np.float64),
                np.asarray(self.triangles, dtype=np.int64), np.asarray(self.speed, dtype=np.float64))

    @property
    def length(self) -> float:
        p = np.asarray(self.points).reshape(-1, 3)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def trace_single(mesh: SurfaceMesh, D_at_time: Callable[[float], np.ndarray], tri: int, a: float, b: float,
                 h: float, n_steps: int, direction: float = 1.0, integrator: str = "euler",
                 t0: float = 0.0, verbose: bool = False, progress=None) -> Trajectory:
    """Integrate one tracer for ``n_steps`` steps of size ``h`` (time units).
    ``D_at_time(t)`` gives the corner-vector field at the absolute time t."""
    p = Particles(np.array([tri]), np.array([a]), np.array([b]))
    traj = Trajectory()
    time = t0
    pos = p.positions(mesh)[0]
    traj.points.append(pos.copy())
    traj.times.append(time)
    traj.triangles.append(int(p.tri[0]))
    report_every = max(n_steps // 100, 1)
    for i in range(n_steps):
        if progress is not None and i % report_every == 0:
            if progress(i / n_steps, f"step {i} of {n_steps}, t = {time:.4g}") is False:
                raise RunCancelled("cancelled")
        res = advect_step(mesh, lambda f: D_at_time(time + f * h * direction), p, h, direction, integrator)
        time += res["used"][0] * direction
        if i == 0:
            traj.speed.append(float(res["speed"][0]))
        traj.speed.append(float(res["speed"][0]))
        traj.points.append(p.positions(mesh)[0].copy())
        traj.times.append(time)
        traj.triangles.append(int(p.tri[0]))
        if p.status[0] != ACTIVE:
            traj.status = int(p.status[0])
            traj.end_reason = STATUS_NAMES[int(p.status[0])]
            if verbose:
                print(f"  tracer stopped after {i + 1} steps: {traj.end_reason}")
            break
    else:
        traj.end_reason = "integration time reached"
    traj.speed = traj.speed[:len(traj.points)]
    while len(traj.speed) < len(traj.points):
        traj.speed.append(traj.speed[-1] if traj.speed else 0.0)
    return traj
