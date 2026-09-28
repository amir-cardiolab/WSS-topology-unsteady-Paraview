"""
Batch advection of surface tracers (flag_code 1, 2, 5, 6, 7, 9, 10):
residence time (RT), WSS exposure time (WSSET), tracer snapshots, staggered
release, wall-normal diffusion / divergence models and tagged tracers; and
the tracking of the fixed points of an unsteady field (flag_code 8).
"""
from __future__ import annotations

import csv
import os
import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import io_vtk
from .field import FieldSequence, SurfaceVectorField
from .fixedpoints import SADDLE, TYPE_NAMES, find_fixed_points, fixed_points_polydata
from .mesh import SurfaceMesh
from .params import Parameters
from .tracer import (ACTIVE, ERROR, LEFT, OUT_NORMAL, STATUS_NAMES, STUCK, WAITING, Particles, RunCancelled,
                     advect_step)

MODE_SUFFIX = {1: "", 2: "", 5: "stag", 6: "dif", 7: "div", 9: "sdif", 10: "tag"}
SEED_VERTEX, SEED_CENTROID, SEED_FILE = 0, 1, 2


@dataclass
class SeedSet:
    particles: Particles
    kind: np.ndarray          # SEED_VERTEX / SEED_CENTROID / SEED_FILE
    index: np.ndarray         # vertex id, triangle id or point id in the seed file
    copy: np.ndarray          # release number (staggered release)
    release: np.ndarray       # release time of every tracer
    tags: Optional[np.ndarray] = None


def build_seeds(params: Parameters, mesh: SurfaceMesh, verbose: bool = True) -> SeedSet:
    code = int(params.flag_code)
    seed_points = getattr(params, "seed_points", None)
    if code == 10 and seed_points is not None:
        pts = np.asarray(seed_points, dtype=np.float64).reshape(-1, 3)
        seed_tags = getattr(params, "seed_tags", None)
        tags = (np.asarray(seed_tags, dtype=np.int64).reshape(-1) if seed_tags is not None
                else np.zeros(len(pts), dtype=np.int64))
        tri, a, b, proj, dist = mesh.locate(pts)
        if verbose:
            print(f"  {len(pts)} tagged seeds (maximum distance to the surface {dist.max():.3g})")
        base = Particles(tri, a, b)
        kind = np.full(len(pts), SEED_FILE)
        index = np.arange(len(pts))
    elif code == 10:
        path = params.infile_tag
        if not os.path.exists(path) and os.path.exists(path + params.infile_suffix):
            path = path + params.infile_suffix
        data = io_vtk.read_dataset(path)
        pts = io_vtk.polydata_points(data)
        arr = data.GetPointData().GetArray("tracer_tag")
        if arr is None:
            for i in range(data.GetPointData().GetNumberOfArrays()):
                cand = data.GetPointData().GetArray(i)
                if cand is not None and cand.GetNumberOfComponents() == 1:
                    arr = cand
                    break
        from vtkmodules.util import numpy_support as vnp
        tags = (vnp.vtk_to_numpy(arr).astype(np.int64).reshape(-1) if arr is not None
                else np.zeros(len(pts), dtype=np.int64))
        tri, a, b, proj, dist = mesh.locate(pts)
        if verbose:
            print(f"  {len(pts)} tagged seeds read from {path} (max distance to the surface "
                  f"{dist.max():.3g})")
        base = Particles(tri, a, b)
        kind = np.full(len(pts), SEED_FILE)
        index = np.arange(len(pts))
    else:
        tri, a, b = mesh.vertex_seeds()
        kind = np.full(mesh.n_points, SEED_VERTEX)
        index = np.arange(mesh.n_points)
        tags = None
        if int(params.HighRes_flag) == 1:
            tc, ac, bc = mesh.centroid_seeds()
            tri, a, b = np.concatenate([tri, tc]), np.concatenate([a, ac]), np.concatenate([b, bc])
            kind = np.concatenate([kind, np.full(mesh.n_tris, SEED_CENTROID)])
            index = np.concatenate([index, np.arange(mesh.n_tris)])
        base = Particles(tri, a, b)
    n_copies = max(int(params.Num_stag), 1) if code == 5 else 1
    parts = Particles.concatenate([base.copy() for _ in range(n_copies)])
    copy = np.repeat(np.arange(n_copies), base.n)
    release = copy * float(params.stag_delta)
    parts.status[release > 0] = WAITING
    return SeedSet(parts, np.tile(kind, n_copies), np.tile(index, n_copies), copy, release,
                   None if tags is None else np.tile(tags, n_copies))


def _n_steps(params: Parameters) -> int:
    return max(int(np.floor(float(params.Integration_time) / float(params.time_step) + 1e-9)), 1)


class _Snapshots:
    """Writes tracer snapshots (positions of the tracers still in the domain)."""

    def __init__(self, params: Parameters, mesh: SurfaceMesh, seeds: SeedSet, suffix: str, write_files: bool = True):
        self.p, self.mesh, self.seeds, self.suffix = params, mesh, seeds, suffix
        self.write_files = write_files
        self.k = 0
        self.files: List[str] = []
        self.records: List[Dict] = []          # in-memory copy of every snapshot (for the GUI)
        self.lines: Optional[List[List[np.ndarray]]] = ([[] for _ in range(seeds.particles.n)]
                                                        if int(params.write_trajectory_lines) else None)
        self.line_times: List[float] = []

    def write(self, time: float, RT: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        P = self.seeds.particles
        inside = P.inside
        pos = P.positions(self.mesh)
        ids = np.nonzero(inside)[0]
        arrays = {"particle_id": ids.astype(np.int32), "seed_kind": self.seeds.kind[ids].astype(np.int32),
                  "seed_index": self.seeds.index[ids].astype(np.int32), "release": self.seeds.copy[ids].astype(np.int32),
                  "RT": RT[ids], "status": P.status[ids].astype(np.int32)}
        if self.seeds.tags is not None:
            arrays["tracer_tag"] = self.seeds.tags[ids].astype(np.int32)
        if y is not None:
            arrays["y_normal"] = y[ids]
        self.records.append({"time": float(time), "positions": pos[ids].copy(), "ids": ids.copy(), "RT": RT[ids].copy(),
                             "status": P.status[ids].copy(), "tags": None if self.seeds.tags is None else self.seeds.tags[ids].copy(),
                             "y_normal": None if y is None else y[ids].copy()})
        if self.write_files:
            pd = io_vtk.points_polydata(pos[ids], arrays, {"time": np.array([time])})
            path = self.p.output_path(f"Traj{self.suffix}.{self.k}")
            io_vtk.write_polydata(pd, path)
            self.files.append(path)
        self.k += 1
        if self.lines is not None:
            self.line_times.append(time)
            for i in ids:
                self.lines[i].append(pos[i].copy())

    def write_pvd(self) -> Optional[str]:
        """ParaView collection file (.pvd) listing the snapshots as a time series."""
        if not self.files:
            return None
        path = self.p.output_path(f"Traj{self.suffix}", ".pvd")
        folder = os.path.dirname(path)
        with open(path, "w") as fh:
            fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
            for rec, f in zip(self.records, self.files):
                fh.write(f'  <DataSet timestep="{rec["time"]:.9g}" group="" part="0" file="{os.path.relpath(f, folder)}"/>\n')
            fh.write('</Collection>\n</VTKFile>\n')
        return path

    def write_lines(self) -> Optional[str]:
        if self.lines is None or not self.write_files:
            return None
        polys = [np.asarray(l) for l in self.lines if len(l) >= 2]
        ids = [i for i, l in enumerate(self.lines) if len(l) >= 2]
        if not polys:
            return None
        cell_arrays = {"particle_id": np.asarray(ids, dtype=np.int32),
                       "seed_index": self.seeds.index[ids].astype(np.int32)}
        if self.seeds.tags is not None:
            cell_arrays["tracer_tag"] = self.seeds.tags[ids].astype(np.int32)
        pd = io_vtk.polylines_polydata(polys, None, cell_arrays)
        path = self.p.output_path(f"TrajLines{self.suffix}")
        io_vtk.write_polydata(pd, path)
        return path


def run_advection(params: Parameters, mesh: SurfaceMesh, seq: FieldSequence, verbose: bool = True,
                  progress=None, write_files: bool = True) -> Dict:
    """Advect all tracers; compute RT and WSSET; write the outputs.

    ``progress(fraction, message)`` is called regularly; returning False cancels
    the run (RunCancelled is raised).  With ``write_files=False`` nothing is
    written and the results are only returned in ``summary["data"]``."""
    code = int(params.flag_code)
    suffix = MODE_SUFFIX.get(code, "")
    h = float(params.time_step)
    n_steps = _n_steps(params)
    T_total = n_steps * h
    direction = params.direction
    out_freq = max(n_steps // max(int(params.number_output_files), 1), 1)
    seeds = build_seeds(params, mesh, verbose)
    P = seeds.particles
    n = P.n
    RT = np.zeros(n)
    ET = np.zeros(mesh.n_tris)
    normal_model = code in (6, 7, 9)
    mu = float(params.mu)
    y_ref = float(params.WSS_SCALE) * mu               # = delta_n, the distance of the tracers
    y = None
    if normal_model:
        y = np.zeros(n) if code == 9 else np.full(n, y_ref)
    rng = np.random.default_rng(int(params.random_seed) or None)
    steady = seq.steady
    D_const = seq.corner_vectors(0.0) if steady else None
    snaps = _Snapshots(params, mesh, seeds, suffix, write_files)
    report_every = max(n_steps // 200, 1)
    if verbose:
        print(f"  {n} tracers, {n_steps} steps of {h:g} (T = {T_total:g}), "
              f"{'backward' if direction < 0 else 'forward'} integration, {params.integrator}")
    time = 0.0
    t_wall = _time.time()
    next_report = 0.1
    for i in range(n_steps):
        if code == 5:
            rel = (P.status == WAITING) & (seeds.release <= time + 1e-12)
            if np.any(rel):
                P.status[rel] = ACTIVE
                if verbose:
                    print(f"  t = {time:.4g}: {int(rel.sum())} tracers released")
        if i == 0 or i % out_freq == 0:
            snaps.write(time, RT, y)
        if steady:
            D_at = lambda f: D_const
        else:
            D_at = lambda f, t=time: seq.corner_vectors(t + f * h)
        speed_scale = None if y is None else y / y_ref
        stuck_before = P.status == STUCK
        res = advect_step(mesh, D_at, P, h, direction, params.integrator, et_accum=ET,
                          speed_scale=speed_scale, max_rounds=int(params.max_crossings_per_step))
        RT += res["used"]
        if np.any(stuck_before):
            RT[stuck_before] += h
            np.add.at(ET, P.tri[stuck_before], h)
        if normal_model:
            inside = P.inside
            if code in (6, 9):
                y[inside] += rng.normal(0.0, np.sqrt(2.0 * float(params.Diff_coef) * h), int(inside.sum()))
            else:
                if params.infile_div:
                    div_nodal = seq.point_scalars(params.infile_div, "wss_div", time)
                    alpha = P.bary(mesh)
                    div_p = np.einsum("nj,nj->n", alpha, div_nodal[mesh.triangles[P.tri]])
                else:
                    ft = seq.field_at_time(time) if not steady else seq.field_at_index(seq.index_first)
                    div_p = ft.divergence()[P.tri] / float(params.WSS_SCALE)
                y[inside] += direction * (-div_p[inside]) * h * y[inside] ** 2 / (2.0 * mu)
            y[:] = np.abs(y)
            out = inside & (y > float(params.max_yn))
            P.status[out] = OUT_NORMAL
        time += h
        if progress is not None and (i % report_every == 0 or i == n_steps - 1):
            act = int(np.sum(P.status == ACTIVE))
            if progress((i + 1) / n_steps, f"t = {time:.4g} of {T_total:g}, {act} active tracers") is False:
                raise RunCancelled("cancelled")
        if verbose and (i + 1) / n_steps >= next_report:
            act = int(np.sum(P.status == ACTIVE))
            print(f"  t = {time:.4g} ({100 * (i + 1) / n_steps:.0f}%), {act} active tracers, "
                  f"{_time.time() - t_wall:.1f} s")
            next_report += 0.1
    snaps.write(time, RT, y)
    line_file = snaps.write_lines()
    pvd_file = snaps.write_pvd()
    # ---- outputs ---------------------------------------------------------
    counts = {STATUS_NAMES[k]: int(v) for k, v in zip(*np.unique(P.status, return_counts=True))}
    files: Dict[str, object] = {"Traj": snaps.files}
    if line_file:
        files["TrajLines"] = line_file
    if pvd_file:
        files["Traj_pvd"] = pvd_file
    wsset = ET / T_total * np.sqrt(mesh.mean_area / mesh.area)
    RT_vertex = np.full(mesh.n_points, np.nan)
    vert = seeds.kind == SEED_VERTEX
    if np.any(vert):
        acc = np.zeros(mesh.n_points)
        cnt = np.zeros(mesh.n_points)
        np.add.at(acc, seeds.index[vert], RT[vert])
        np.add.at(cnt, seeds.index[vert], 1.0)
        RT_vertex = np.divide(acc, cnt, out=np.full(mesh.n_points, np.nan), where=cnt > 0)
    data = {"snapshots": snaps.records, "RT": RT, "RT_vertex": RT_vertex, "ET": ET, "WSSET": wsset,
            "seed_kind": seeds.kind, "seed_index": seeds.index, "seed_copy": seeds.copy, "tags": seeds.tags,
            "status": P.status.copy(), "y_normal": y, "mesh": mesh}
    # residence time on the seeds
    if write_files and code in (1, 5, 6, 7, 9, 10):
        vert = seeds.kind == SEED_VERTEX
        point_arrays, cell_arrays = {}, {}
        if np.any(vert):
            RT_v = np.zeros(mesh.n_points)
            cnt = np.zeros(mesh.n_points)
            np.add.at(RT_v, seeds.index[vert], RT[vert])
            np.add.at(cnt, seeds.index[vert], 1.0)
            point_arrays["RT"] = np.divide(RT_v, cnt, out=np.zeros_like(RT_v), where=cnt > 0)
            first = vert & (seeds.copy == 0)
            st = np.zeros(mesh.n_points, dtype=np.int32)
            st[seeds.index[first]] = P.status[first]
            point_arrays["status"] = st
            if y is not None:
                yv = np.zeros(mesh.n_points)
                yv[seeds.index[first]] = y[first]
                point_arrays["y_normal"] = yv
        cen = seeds.kind == SEED_CENTROID
        if np.any(cen):
            RT_c = np.zeros(mesh.n_tris)
            cnt = np.zeros(mesh.n_tris)
            np.add.at(RT_c, seeds.index[cen], RT[cen])
            np.add.at(cnt, seeds.index[cen], 1.0)
            cell_arrays["RT_centroid"] = np.divide(RT_c, cnt, out=np.zeros_like(RT_c), where=cnt > 0)
        if code == 10:
            pd = io_vtk.points_polydata(P.positions(mesh), {"RT": RT, "tracer_tag": seeds.tags.astype(np.int32),
                                                             "status": P.status.astype(np.int32)})
            files["RT_tracers"] = io_vtk.write_polydata(pd, params.output_path(f"RT{suffix}_tracers"))
        if point_arrays or cell_arrays:
            pd = mesh.to_polydata(point_arrays, cell_arrays, {"integration_time": np.array([T_total])})
            files["RT"] = io_vtk.write_polydata(pd, params.output_path(f"RT{suffix}"))
        # exposure time
        pd = mesh.to_polydata(None, {"ET": ET, "ET_norm": ET / np.sqrt(mesh.area), "WSSET": wsset},
                              {"integration_time": np.array([T_total]), "n_tracers": np.array([n]),
                               "mean_area": np.array([mesh.mean_area])})
        pt = io_vtk.cell_to_point(pd, ["WSSET", "ET"])
        pd.GetPointData().AddArray(io_vtk.numpy_to_vtk_array(pt["WSSET"], "WSSET_point"))
        pd.GetPointData().AddArray(io_vtk.numpy_to_vtk_array(pt["ET"], "ET_point"))
        files["ET"] = io_vtk.write_polydata(pd, params.output_path(f"ET{suffix}"))
    summary = {"n_tracers": n, "n_steps": n_steps, "integration_time": T_total, "status": counts,
               "RT_mean": float(RT.mean()), "ET_total": float(ET.sum()), "files": files,
               "wall_time_s": _time.time() - t_wall, "data": data}
    if verbose:
        print(f"  done in {summary['wall_time_s']:.1f} s; tracer status: {counts}")
        print(f"  mean RT = {RT.mean():.4g}, total exposure time = {ET.sum():.6g} "
              f"(maximum {n * T_total:g})")
    return summary


def run_fixed_point_tracking(params: Parameters, mesh: SurfaceMesh, seq: FieldSequence,
                             verbose: bool = True, progress=None, write_files: bool = True) -> Dict:
    """flag_code 8: fixed points of the time-interpolated field at every time
    step, their exposure time per element and per node."""
    h = float(params.time_step)
    n_steps = _n_steps(params)
    T_total = n_steps * h
    out_freq = max(n_steps // max(int(params.number_output_files), 1), 1)
    sing_et = np.zeros(mesh.n_tris)
    nodal = np.zeros(mesh.n_points)
    nodal_eig = np.zeros(mesh.n_points)
    counts_rows = []
    files: Dict[str, object] = {"SingTraj": []}
    steady = seq.steady
    t_wall = _time.time()
    time = 0.0
    k_out = 0
    fps = []
    field_t = None
    records: List[Dict] = []
    report_every = max(n_steps // 200, 1)
    for i in range(n_steps):
        if progress is not None and (i % report_every == 0 or i == n_steps - 1):
            if progress((i + 1) / n_steps, f"t = {time:.4g} of {T_total:g}, {len(fps)} fixed points") is False:
                raise RunCancelled("cancelled")
        if field_t is None or not steady:
            field_t = seq.field_at_time(time)
            fps, _ = find_fixed_points(field_t, bool(params.exclude_boundary_fixed_points),
                                       float(params.zero_vector_tolerance))
        for fp in fps:
            t = fp.triangle
            sing_et[t] += h
            verts = mesh.triangles[t]
            d = np.linalg.norm(mesh.points[verts] - fp.position, axis=1)
            den = d.sum()
            w = (den - d) / den if den > 0 else np.full(3, 2.0 / 3.0)   # weights of the original code
            nodal[verts] += w * h
            nodal_eig[verts] += w * h * fp.eigenvalue_total
        if i == 0 or i % out_freq == 0:
            records.append({"time": float(time), "fixed_points": list(fps)})
            if write_files:
                pd = fixed_points_polydata(fps, float(params.WSS_SCALE))
                pd.GetFieldData().AddArray(io_vtk.numpy_to_vtk_array(np.array([time]), "time"))
                path = params.output_path(f"SingTraj.{k_out}")
                io_vtk.write_polydata(pd, path)
                files["SingTraj"].append(path)
            k_out += 1
        row = {"time": time, "n_fixed_points": len(fps)}
        for code, name in TYPE_NAMES.items():
            row["n_" + name.replace(" ", "_")] = sum(1 for fp in fps if fp.type_code == code)
        counts_rows.append(row)
        time += h
        if verbose and (i + 1) % max(n_steps // 10, 1) == 0:
            print(f"  t = {time:.4g} ({100 * (i + 1) / n_steps:.0f}%), {len(fps)} fixed points, "
                  f"{_time.time() - t_wall:.1f} s")
    if write_files:
        pd = mesh.to_polydata({"SingET_nodal": nodal, "SingET_nodal_eig": nodal_eig},
                              {"SingET": sing_et, "SingET_norm": sing_et / np.sqrt(mesh.area)},
                              {"integration_time": np.array([T_total])})
        files["SingET"] = io_vtk.write_polydata(pd, params.output_path("SingET"))
        csv_path = params.output_path("SingCount", ".csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(counts_rows[0].keys()))
            w.writeheader()
            w.writerows(counts_rows)
        files["SingCount"] = csv_path
        if files["SingTraj"]:
            pvd = params.output_path("SingTraj", ".pvd")
            with open(pvd, "w") as fh:
                fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
                for rec, f in zip(records, files["SingTraj"]):
                    fh.write(f'  <DataSet timestep="{rec["time"]:.9g}" group="" part="0" file="{os.path.relpath(f, os.path.dirname(pvd))}"/>\n')
                fh.write('</Collection>\n</VTKFile>\n')
            files["SingTraj_pvd"] = pvd
    summary = {"n_steps": n_steps, "integration_time": T_total, "files": files,
               "mean_fixed_points": float(np.mean([r["n_fixed_points"] for r in counts_rows])),
               "wall_time_s": _time.time() - t_wall,
               "data": {"snapshots": records, "SingET": sing_et, "SingET_nodal": nodal, "SingET_nodal_eig": nodal_eig,
                        "counts": counts_rows, "mesh": mesh}}
    if verbose:
        print(f"  done in {summary['wall_time_s']:.1f} s; on average {summary['mean_fixed_points']:.2f} "
              f"fixed points per time step")
    return summary
