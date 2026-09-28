"""
Command line driver:  python -m wsslcs input.in
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from typing import Dict, Optional

import numpy as np

from . import io_vtk
from .api import analyze, write_results
from .exposure import run_advection, run_fixed_point_tracking
from .metrics import run_metrics
from .field import FieldSequence, SurfaceVectorField
from .mesh import SurfaceMesh
from .params import ADVECTION_CODES, FLAG_CODES, Parameters, write_template
from .tracer import trace_single


def _single_tracer(params: Parameters, mesh: SurfaceMesh, seq: FieldSequence, verbose: bool,
                   progress=None, write_files: bool = True) -> Dict:
    code = int(params.flag_code)
    if code == 4:
        idx = int(params.pt_IC_index)
        if idx < 0 or idx >= mesh.n_points:
            raise ValueError(f"pt_IC_index = {idx} is outside 0..{mesh.n_points - 1}")
        tri, a, b = mesh.vertex_seeds([idx])
        start = mesh.points[idx]
        if verbose:
            print(f"  tracer released at vertex {idx}: {start}")
    else:
        pt = np.asarray(params.pt_IC, dtype=np.float64)
        tri, a, b, proj, dist = mesh.locate(pt[None, :])
        start = proj[0]
        if verbose:
            print(f"  tracer released at {pt} (snapped to the surface point {start}, distance {dist[0]:.3g})")
    h = float(params.time_step)
    n_steps = max(int(np.floor(float(params.Integration_time) / h + 1e-9)), 1)
    steady = seq.steady
    D0 = seq.corner_vectors(0.0) if steady else None
    D_at_time = (lambda t: D0) if steady else (lambda t: seq.corner_vectors(abs(t)))
    t0 = _time.time()
    traj = trace_single(mesh, D_at_time, int(tri[0]), float(a[0]), float(b[0]), h, n_steps,
                        params.direction, params.integrator, verbose=verbose, progress=progress)
    pts, times, tris, speed = traj.as_arrays()
    files = {}
    if write_files:
        pd = io_vtk.polylines_polydata([pts], {"time": [times], "wss_magnitude": [speed / params.WSS_SCALE],
                                                "triangle": [tris.astype(np.int32)]},
                                        {"end_reason": [traj.end_reason], "length": [traj.length]})
        files["Traj"] = io_vtk.write_polydata(pd, params.output_path("Traj"))
    if verbose:
        print(f"  {len(pts)} points, path length {traj.length:.4g}, {traj.end_reason}; "
              f"{'written ' + files['Traj'] if files else 'not written'} ({_time.time() - t0:.1f} s)")
    return {"files": files, "n_points": len(pts), "length": traj.length, "end_reason": traj.end_reason,
            "data": {"trajectory": traj, "start": np.asarray(start, dtype=np.float64)}}


def _topology(params: Parameters, mesh: SurfaceMesh, seq: FieldSequence, verbose: bool,
              progress=None, write_files: bool = True) -> Dict:
    code = int(params.flag_code)
    if seq.steady:
        fld = seq.field_at_index(seq.index_first)
    else:
        if verbose:
            print("  computing the time averaged WSS field (TAWSS)")
        fld = seq.time_average()
    res = analyze(fld, wss_scale=params.WSS_SCALE, manifolds=(code == 12),
                  exclude_boundary=bool(params.exclude_boundary_fixed_points),
                  zero_tol=float(params.zero_vector_tolerance), step_fraction=float(params.manifold_step_fraction),
                  max_steps=int(params.manifold_max_steps), max_length=float(params.manifold_max_length),
                  perturbation=float(params.manifold_perturbation), capture_radius=float(params.fixed_point_capture_radius),
                  max_crossings=int(params.max_crossings_per_step), verbose=verbose)
    files = write_results(res, params.output_dir, params.output_prefix, params.output_format) if write_files else {}
    if verbose:
        print(res.summary())
        if files:
            print("  written: " + ", ".join(files.values()))
    return {"files": files, "n_fixed_points": len(res.fixed_points), "n_branches": len(res.branches),
            "timings": res.timings, "data": {"result": res}}


def run(params: Parameters, verbose: Optional[bool] = None, progress=None, write_files: bool = True) -> Dict:
    """Run the computation selected by ``flag_code``.

    Returns a dictionary with the summary (written to ``<prefix>_summary.json``
    when files are written) and, under ``"data"``, the in-memory results
    (mesh, field, fixed points, manifolds, tracer snapshots, ...)."""
    verbose = bool(params.verbose) if verbose is None else verbose
    params.verbose = int(verbose)
    params.validate()
    code = int(params.flag_code)
    t0 = _time.time()
    if verbose:
        print(f"WSSLCS: flag_code = {code} ({FLAG_CODES[code]})")
        print(f"  reading {params.data_file()}")
    seq = FieldSequence(params)
    mesh = seq.mesh
    if verbose:
        print("  " + mesh.summary())
        if not seq.steady:
            print("  " + seq.describe(params.Integration_time if code not in (11, 12, 13) else None))
    if code == 13:
        out = run_metrics(params, mesh, seq, verbose, progress, write_files)
    elif code in (3, 4):
        out = _single_tracer(params, mesh, seq, verbose, progress, write_files)
    elif code in (11, 12):
        out = _topology(params, mesh, seq, verbose, progress, write_files)
    elif code == 8:
        out = run_fixed_point_tracking(params, mesh, seq, verbose, progress, write_files)
    elif code in ADVECTION_CODES:
        out = run_advection(params, mesh, seq, verbose, progress, write_files)
    else:
        raise ValueError(f"unsupported flag_code {code}")
    out["flag_code"] = code
    out["total_time_s"] = _time.time() - t0
    data = out.pop("data", {})
    data.setdefault("mesh", mesh)
    data.setdefault("field", seq.field_at_index(seq.index_first))
    data["sequence"] = seq
    if write_files:
        with open(params.output_path("summary", ".json"), "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    out["data"] = data
    if verbose:
        print(f"Done in {out['total_time_s']:.1f} s")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wsslcs", description="Lagrangian analysis of WSS vector fields on surfaces")
    ap.add_argument("input", nargs="?", help="parameter file (VARIABLE = VALUE format)")
    ap.add_argument("--template", metavar="FILE", help="write a template parameter file and exit")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a parameter of the input file (may be repeated)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.template:
        write_template(args.template)
        print(f"template written to {args.template}")
        return 0
    if not args.input:
        ap.error("an input file is required (or --template FILE)")
    params = Parameters.from_file(args.input)
    if args.set:
        params.update(dict(s.split("=", 1) for s in args.set))
        params.validate()
    run(params, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
