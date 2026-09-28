"""
High level interface: fixed points + stable/unstable manifolds of a surface
vector field (used by the command line driver, the web application and
scripts / notebooks).
"""
from __future__ import annotations

import csv
import os
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
try:
    import vtk
except ImportError:                                  # ParaView's Python: no top-level 'vtk' module
    import vtkmodules.all as vtk

from . import io_vtk
from .field import SurfaceVectorField
from .fixedpoints import (FixedPoint, SADDLE, TYPE_NAMES, find_fixed_points, fixed_points_polydata,
                          summarize)
from .manifolds import ManifoldBranch, compute_manifolds, manifolds_polydata
from .mesh import SurfaceMesh


@dataclass
class AnalysisResult:
    mesh: SurfaceMesh
    field: SurfaceVectorField
    fixed_points: List[FixedPoint]
    poincare_index: np.ndarray
    branches: List[ManifoldBranch]
    vector_name: str
    scale: float
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def saddles(self) -> List[FixedPoint]:
        return [fp for fp in self.fixed_points if fp.type_code == SADDLE]

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for fp in self.fixed_points:
            counts[fp.type_name] = counts.get(fp.type_name, 0) + 1
        lines = [self.mesh.summary(),
                 f"vector array '{self.vector_name}', scale {self.scale:g}, "
                 f"|v| range {self.field.mag.min():.4g} .. {self.field.mag.max():.4g}",
                 f"{len(self.fixed_points)} fixed points: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
                 f"sum of the Poincare indices: {int(self.poincare_index.sum())}",
                 f"{len(self.branches)} manifold branches, total length "
                 f"{sum(br.length for br in self.branches):.4g}"]
        if self.fixed_points:
            lines.append(summarize(self.fixed_points))
        return "\n".join(lines)


def analyze(source: Union[str, vtk.vtkPolyData, SurfaceVectorField], vector_array: str = "wss",
            wss_scale: float = 1.0, manifolds: bool = True, exclude_boundary: bool = True,
            zero_tol: float = 1e-10, step_fraction: float = 0.2, max_steps: int = 20000,
            max_length: float = 0.0, perturbation: float = 0.1, capture_radius: float = 0.5,
            max_crossings: int = 50, verbose: bool = False) -> AnalysisResult:
    """Fixed points and (optionally) manifolds of the field in ``source``
    (a file name, a vtkPolyData with a point vector array, or a
    SurfaceVectorField)."""
    timings = {}
    t0 = _time.time()
    if isinstance(source, SurfaceVectorField):
        fld = source
        used = fld.name
    else:
        if isinstance(source, str):
            pts, tris, vec, used, _ = io_vtk.read_surface(source, vector_array)
        else:
            pd = io_vtk.as_triangle_polydata(source)
            pts, tris = io_vtk.polydata_points(pd), io_vtk.polydata_triangles(pd)
            vec, used = io_vtk.get_point_vectors(pd, vector_array)
        mesh = SurfaceMesh(pts, tris)
        fld = SurfaceVectorField(mesh, vec, scale=wss_scale, name=used or vector_array)
    timings["load"] = _time.time() - t0
    t0 = _time.time()
    fps, index = find_fixed_points(fld, exclude_boundary, zero_tol, verbose=verbose)
    timings["fixed_points"] = _time.time() - t0
    t0 = _time.time()
    branches = []
    if manifolds:
        branches = compute_manifolds(fld.mesh, fld, fps, step_fraction, max_steps, max_length or None,
                                     perturbation, capture_radius, max_crossings, verbose=verbose)
    timings["manifolds"] = _time.time() - t0
    return AnalysisResult(fld.mesh, fld, fps, index, branches, used or vector_array, wss_scale, timings)


def surface_polydata(result: AnalysisResult) -> vtk.vtkPolyData:
    """The surface with the tangential (scaled) WSS, its magnitude, the
    Poincare index and the divergence of every triangle."""
    fld, mesh = result.field, result.mesh
    s = result.scale if result.scale else 1.0
    return mesh.to_polydata(
        {"wss_tangential": fld.t_vec / s, "wss_magnitude": fld.mag / s, "velocity": fld.t_vec},
        {"poincare_index": result.poincare_index.astype(np.int32), "divergence": fld.divergence() / s,
         "area": mesh.area, "boundary": mesh.boundary_triangle.astype(np.int32)})


def write_results(result: AnalysisResult, output_dir: str, prefix: str, fmt: str = "vtp") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ext = "." + fmt.lstrip(".")
    files = {}
    files["FixedPoints"] = io_vtk.write_polydata(fixed_points_polydata(result.fixed_points, result.scale),
                                                 os.path.join(output_dir, f"{prefix}_FixedPoints{ext}"))
    csv_path = os.path.join(output_dir, f"{prefix}_FixedPoints.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type", "type_name", "poincare_index", "triangle", "x", "y", "z",
                    "eig1_real", "eig1_imag", "eig2_real", "eig2_imag", "residual"])
        for fp in result.fixed_points:
            e = fp.eigenvalues / result.scale
            w.writerow([fp.id, fp.type_code, fp.type_name, fp.poincare_index, fp.triangle, *fp.position,
                        e[0].real, e[0].imag, e[1].real, e[1].imag, fp.residual])
    files["FixedPoints_csv"] = csv_path
    if result.branches:
        files["Manifolds"] = io_vtk.write_polydata(manifolds_polydata(result.branches, result.scale),
                                                   os.path.join(output_dir, f"{prefix}_Manifolds{ext}"))
    files["Surface"] = io_vtk.write_polydata(surface_polydata(result), os.path.join(output_dir, f"{prefix}_Surface{ext}"))
    return files


def result_to_dict(result: AnalysisResult, decimals: int = 5) -> Dict:
    """JSON-serialisable representation used by the web application."""
    mesh, fld = result.mesh, result.field
    s = result.scale if result.scale else 1.0
    r = lambda a: np.round(np.asarray(a, dtype=np.float64), decimals).ravel().tolist()
    fps = []
    for fp in result.fixed_points:
        fps.append({"id": fp.id, "type": fp.type_code, "type_name": fp.type_name, "index": fp.poincare_index,
                    "triangle": fp.triangle, "position": r(fp.position),
                    "eigenvalues": [[float(e.real / s), float(e.imag / s)] for e in fp.eigenvalues],
                    "eigvec_out": r(fp.eigvec_out) if fp.eigvec_out is not None else None,
                    "eigvec_in": r(fp.eigvec_in) if fp.eigvec_in is not None else None,
                    "residual": float(fp.residual / s)})
    branches = []
    for br in result.branches:
        branches.append({"saddle": br.saddle, "kind": br.kind, "kind_name": br.kind_name, "sign": br.sign,
                         "points": r(br.points), "length": float(br.length), "end_reason": br.end_reason,
                         "end_fixed_point": br.end_fixed_point})
    return {
        "n_points": mesh.n_points, "n_triangles": mesh.n_tris,
        "points": r(mesh.points), "triangles": mesh.triangles.ravel().tolist(),
        "magnitude": r(fld.mag / s), "vectors": r(fld.t_vec / s),
        "bounds": r(mesh.bounds), "center": r(mesh.center), "radius": float(mesh.radius),
        "mean_edge_length": float(mesh.mean_edge_length),
        "fixed_points": fps, "branches": branches,
        "poincare_index_sum": int(result.poincare_index.sum()),
        "vector_name": result.vector_name, "scale": result.scale, "timings": result.timings,
        "type_names": {int(k): v for k, v in TYPE_NAMES.items()},
    }
