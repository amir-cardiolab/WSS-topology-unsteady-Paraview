"""
Time-series (Eulerian) WSS metrics computed over one period of an unsteady
WSS data set (flag_code 13):

* TAWSS          time-averaged WSS magnitude            (1/T) int |tau| dt
* TAWSS_vector   time-averaged WSS vector               (1/T) int  tau  dt
* OSI            oscillatory shear index                0.5 (1 - |int tau dt| / int |tau| dt)
* RRT            relative residence time                1 / |(1/T) int tau dt|   (= 1 / ((1 - 2 OSI) TAWSS))
* WSSdiv         time-averaged WSS divergence           (1/T) int  div(tau) dt         (Eq. 3 of the WSSET paper)
* DIVW_mean      time-averaged divergence of the unit WSS field   DIV_W = div(tau / |tau|)
* TSVI           topological shear variation index      [ (1/T) int (DIV_W - mean DIV_W)^2 dt ]^(1/2)
                 (Mazzi et al., Biomech. Model. Mechanobiol. 2020)

The divergences are surface divergences of the piecewise-linear field in
every triangle (cell values) and are also averaged to the points (area
weighted).  The frames of one period are equally spaced, so the time
integrals are arithmetic means over the frames.  With steady (single file)
data the metrics degenerate (OSI = TSVI = 0).
"""
from __future__ import annotations

import time as _time
from typing import Dict, Tuple

import numpy as np

from . import io_vtk
from .field import FieldSequence, SurfaceVectorField
from .mesh import SurfaceMesh
from .params import Parameters
from .tracer import RunCancelled


def compute_metrics(seq: FieldSequence, progress=None, verbose: bool = False) -> Dict[str, Tuple[np.ndarray, str]]:
    """name -> (values, 'point' | 'cell'); the WSS is expressed in the units of the files."""
    mesh = seq.mesh
    scale = float(seq.p.WSS_SCALE) or 1.0
    idxs = [seq.index_first] if seq.steady else seq.frame_list()
    n = len(idxs)
    sum_vec = np.zeros((mesh.n_points, 3))
    sum_mag = np.zeros(mesh.n_points)
    sum_div = np.zeros(mesh.n_tris)
    sum_divw = np.zeros(mesh.n_tris)
    sum_divw2 = np.zeros(mesh.n_tris)
    for k, idx in enumerate(idxs):
        if progress is not None and progress(k / n, f"frame {k + 1} of {n} (file index {idx})") is False:
            raise RunCancelled("cancelled")
        f = seq.field_at_index(idx)
        tau = f.t_vec / scale
        mag = f.mag / scale
        sum_vec += tau
        sum_mag += mag
        sum_div += f.divergence() / scale
        unit = np.divide(tau, mag[:, None], out=np.zeros_like(tau), where=mag[:, None] > 0)
        fu = SurfaceVectorField(mesh, unit, scale=1.0, name="unit")
        divw = fu.divergence()
        sum_divw += divw
        sum_divw2 += divw ** 2
        if verbose:
            print(f"  frame {k + 1}/{n} (index {idx}) processed")
    mean_vec = sum_vec / n
    tawss = sum_mag / n
    mean_vec_mag = np.linalg.norm(mean_vec, axis=1)
    osi = np.where(tawss > 0, 0.5 * (1.0 - np.divide(mean_vec_mag, tawss, out=np.ones_like(tawss), where=tawss > 0)), 0.0)
    osi = np.clip(osi, 0.0, 0.5)
    floor = 1e-12 * max(float(tawss.max()), 1e-300)
    rrt = 1.0 / np.maximum(mean_vec_mag, floor)
    wssdiv = sum_div / n
    divw_mean = sum_divw / n
    tsvi = np.sqrt(np.maximum(sum_divw2 / n - divw_mean ** 2, 0.0))
    out: Dict[str, Tuple[np.ndarray, str]] = {
        "TAWSS": (tawss, "point"),
        "TAWSS_vector": (mean_vec, "point"),
        "TAWSS_vector_magnitude": (mean_vec_mag, "point"),
        "OSI": (osi, "point"),
        "RRT": (rrt, "point"),
        "WSSdiv": (wssdiv, "cell"),
        "WSSdiv_point": (mesh.cell_to_point(wssdiv), "point"),
        "DIVW_mean": (divw_mean, "cell"),
        "DIVW_mean_point": (mesh.cell_to_point(divw_mean), "point"),
        "TSVI": (tsvi, "cell"),
        "TSVI_point": (mesh.cell_to_point(tsvi), "point"),
    }
    if progress is not None:
        progress(1.0, "metrics computed")
    return out


def run_metrics(params: Parameters, mesh: SurfaceMesh, seq: FieldSequence, verbose: bool = True,
                progress=None, write_files: bool = True) -> Dict:
    t0 = _time.time()
    if seq.steady and verbose:
        print("  note: the time-series metrics are computed from a single file (OSI = TSVI = 0)")
    metrics = compute_metrics(seq, progress, verbose)
    n_frames = 1 if seq.steady else seq.n_frames
    period = 0.0 if seq.steady else seq.period
    files = {}
    if write_files:
        point_arrays = {k: v for k, (v, loc) in metrics.items() if loc == "point"}
        cell_arrays = {k: v for k, (v, loc) in metrics.items() if loc == "cell"}
        pd = mesh.to_polydata(point_arrays, cell_arrays, {"n_frames": np.array([n_frames]), "period": np.array([period])})
        files["WSSmetrics"] = io_vtk.write_polydata(pd, params.output_path("WSSmetrics"))
    summary = {"n_frames": n_frames, "period": period, "files": files, "wall_time_s": _time.time() - t0,
               "TAWSS_mean": float(metrics["TAWSS"][0].mean()), "OSI_mean": float(metrics["OSI"][0].mean()),
               "OSI_max": float(metrics["OSI"][0].max()), "TSVI_max": float(metrics["TSVI"][0].max()),
               "WSSdiv_range": [float(metrics["WSSdiv"][0].min()), float(metrics["WSSdiv"][0].max())],
               "data": {"metrics": metrics}}
    if verbose:
        print(f"  {n_frames} frames, period {period:g}: TAWSS mean {summary['TAWSS_mean']:.4g}, OSI mean {summary['OSI_mean']:.4g} "
              f"(max {summary['OSI_max']:.3g}), TSVI max {summary['TSVI_max']:.4g}, "
              f"WSSdiv {summary['WSSdiv_range'][0]:.4g} .. {summary['WSSdiv_range'][1]:.4g}")
        if files:
            print(f"  written {files['WSSmetrics']}")
    return summary
