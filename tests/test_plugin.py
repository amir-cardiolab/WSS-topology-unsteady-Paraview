"""End-to-end test of the ParaView plugin, run with ParaView's Python:

    pvpython tests/test_plugin.py

Loads the plugin, runs the analyses on the example data (a synthetic 8-frame WSS
sequence and a steady WSS patch) through the ParaView pipeline and compares the
outputs with the command-line program (`wsslcs.cli.run`, same package, same files).
"""
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
from paraview import servermanager
from paraview.simple import LegacyVTKReader, LoadPlugin, XMLPolyDataReader
from vtkmodules.util import numpy_support as vnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from wsslcs import cli                              # noqa: E402
from wsslcs.params import Parameters                # noqa: E402

DATA = os.path.join(ROOT, "examples", "data")
FRAMES = [os.path.join(DATA, f"Carotid_patch_synthetic_unsteady-{k}.vtk") for k in range(8)]
PATCH = os.path.join(DATA, "Carotid_patch_TAWSS.vtk")
TAGS = os.path.join(DATA, "Carotid_patch_tags.vtp")
failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
    print(("  ok: " if cond else "  FAIL: ") + msg)


def arr(pd, name, cell=False):
    a = (pd.GetCellData() if cell else pd.GetPointData()).GetArray(name)
    return None if a is None else vnp.vtk_to_numpy(a)


def fetch(proxy, port=0):
    # built-in server: the client side object is the algorithm itself
    return proxy.GetClientSideObject().GetOutputDataObject(port)


def close(name, a, b, tol):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        check(False, f"{name}: shape {a.shape} vs {b.shape}")
        return
    d = np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))) if a.size else 0.0
    check(d <= tol, f"{name}: max |diff| {d:.2e} (n = {a.size}, tol {tol})")


def reference(**kw):
    base = dict(infile=os.path.join(DATA, "Carotid_patch_synthetic_unsteady-"), infile_suffix=".vtk", FILE_index_start=0,
                FILE_index_end=7, index_delta=1, delta_t_file=0.1, index_first=3, Steady_flag=0, WSS_SCALE=0.03,
                output_dir=tempfile.gettempdir(), verbose=0)
    base.update(kw)
    p = Parameters(**base)
    p.validate()
    return cli.run(p, verbose=False, write_files=False)


def main():
    LoadPlugin(os.path.join(ROOT, "WSSLCSPlugin.py"), remote=True, ns=globals())
    Filter = globals()["WSSSurfaceTransport"]
    reader = LegacyVTKReader(FileNames=FRAMES)
    reader.UpdatePipelineInformation()
    check(len(reader.TimestepValues) == 8, f"file series with {len(reader.TimestepValues)} time steps: {list(reader.TimestepValues)}")

    def make(**props):
        f = Filter(Input=reader)
        f.WSSVectors = ["POINTS", "wss"]
        f.WSSScale = 0.03
        f.TimeBetweenFrames = 0.1
        f.ReleaseFrame = 3
        for k, v in props.items():
            setattr(f, k, v)
        return f

    # ---- 1. tracers, residence time and exposure time over 1.25 periods ---------------------
    print("\n=== residence time / WSSET on the periodic sequence (flag 1) ===")
    f = make(Analysis=1, TimeStep=0.01, IntegrationTime=1.0, NumberOfSnapshots=5)
    f.UpdatePipeline()
    ref = reference(flag_code=1, time_step=0.01, Integration_time=1.0, number_output_files=5)
    surf = fetch(f, 0)
    check(surf.GetNumberOfPoints() == 5838 and surf.GetNumberOfCells() == 11430, f"surface {surf.GetNumberOfPoints()} points, {surf.GetNumberOfCells()} cells")
    check(arr(surf, "wss") is not None, "input arrays are passed through (wss)")
    close("RT", arr(surf, "RT"), np.nan_to_num(ref["data"]["RT_vertex"]), 1e-12)
    close("WSSET", arr(surf, "WSSET", cell=True), ref["data"]["WSSET"], 1e-12)
    close("ET", arr(surf, "ET", cell=True), ref["data"]["ET"], 1e-12)
    snaps = ref["data"]["snapshots"]
    # the snapshot times are expressed in the time axis of the input (frames 0..7 -> 0.1 s each):
    # release at frame 3, elapsed time x 10
    f.UpdatePropertyInformation()
    times = list(f.TimestepValues)
    exp_times = sorted(set(round(3.0 + 10.0 * s["time"], 10) for s in snaps))
    check(len(times) == len(exp_times) and np.allclose(times, exp_times), f"tracer output time steps {times}")
    for t_req, k in ((3.0, 0), (13.0, len(snaps) - 1), (9.0, 3)):
        f.UpdatePipeline(t_req)
        tr = fetch(f, 3)
        rec = snaps[k]
        check(tr.GetNumberOfPoints() == len(rec["ids"]), f"snapshot at t = {t_req}: {tr.GetNumberOfPoints()} tracers (reference {len(rec['ids'])})")
        if tr.GetNumberOfPoints() == len(rec["ids"]):
            close(f"positions at t = {t_req}", vnp.vtk_to_numpy(tr.GetPoints().GetData()), rec["positions"], 1e-6)
            close(f"RT at t = {t_req}", arr(tr, "RT"), rec["RT"], 1e-12)
    summary = f.GetClientSideObject().GetSummary()
    check(summary is not None and summary["n_tracers"] == 5838 and summary["n_steps"] == 100, f"summary: {summary and {k: summary[k] for k in ('n_tracers', 'n_steps', 'integration_time')}}")

    # ---- 2. time-series metrics -------------------------------------------------------------
    print("\n=== metrics (flag 13) ===")
    f = make(Analysis=13)
    f.UpdatePipeline()
    ref = reference(flag_code=13)
    surf = fetch(f, 0)
    for name in ("TAWSS", "OSI", "RRT"):
        close(name, arr(surf, name), ref["data"]["metrics"][name][0], 1e-12)
    for name in ("WSSdiv_TA", "NormWSSdiv_TA", "TSVI", "TSVI_valid"):
        close(name, arr(surf, name, cell=True), ref["data"]["metrics"][name][0], 1e-12)

    # ---- 3. fixed points tracked in time --------------------------------------------------
    print("\n=== fixed points in time (flag 8) ===")
    f = make(Analysis=8, TimeStep=0.02, IntegrationTime=0.4, NumberOfSnapshots=4)
    f.UpdatePipeline()
    ref = reference(flag_code=8, time_step=0.02, Integration_time=0.4, number_output_files=4)
    surf = fetch(f, 0)
    close("SingET", arr(surf, "SingET", cell=True), ref["data"]["SingET"], 1e-12)
    close("SingET_nodal", arr(surf, "SingET_nodal"), ref["data"]["SingET_nodal"], 1e-12)
    f.UpdatePipeline(3.0)
    fp = fetch(f, 3)
    rec = ref["data"]["snapshots"][0]
    check(fp.GetNumberOfPoints() == len(rec["fixed_points"]), f"fixed points at t = 3.0: {fp.GetNumberOfPoints()} (reference {len(rec['fixed_points'])})")
    f.UpdatePipeline(6.0)
    fp = fetch(f, 3)
    rec = ref["data"]["snapshots"][-1]
    check(fp.GetNumberOfPoints() == len(rec["fixed_points"]), f"fixed points at t = 6.0 (last snapshot): {fp.GetNumberOfPoints()} (reference {len(rec['fixed_points'])})")

    # ---- 4. fixed points and manifolds: current time step vs time averaged field -------------
    print("\n=== fixed points + manifolds (flag 12) ===")
    f = make(Analysis=12, UseAllTimeSteps=0)
    f.UpdatePipeline(2.0)
    ref = reference(flag_code=12, Steady_flag=1, infile=FRAMES[2])
    fp, lines = fetch(f, 1), fetch(f, 2)
    check(fp.GetNumberOfPoints() == ref["n_fixed_points"] and lines.GetNumberOfLines() == ref["n_branches"],
          f"current time step 2: {fp.GetNumberOfPoints()} fixed points, {lines.GetNumberOfLines()} branches (reference {ref['n_fixed_points']}, {ref['n_branches']})")
    close("fixed point positions", vnp.vtk_to_numpy(fp.GetPoints().GetData()), np.array([q.position for q in ref["data"]["result"].fixed_points]), 1e-9)
    f.UseAllTimeSteps = 1
    f.UpdatePipeline()
    ref = reference(flag_code=12)
    fp, lines = fetch(f, 1), fetch(f, 2)
    check(fp.GetNumberOfPoints() == ref["n_fixed_points"] and lines.GetNumberOfLines() == ref["n_branches"],
          f"time averaged field: {fp.GetNumberOfPoints()} fixed points, {lines.GetNumberOfLines()} branches (reference {ref['n_fixed_points']}, {ref['n_branches']})")
    close("manifold lengths", sorted(arr(lines, "length", cell=True)), sorted(b.length for b in ref["data"]["result"].branches), 1e-9)

    # ---- 5. single tracer ------------------------------------------------------------------
    print("\n=== single tracer (flag 4) ===")
    f = make(Analysis=4, ReleaseVertex=100, TimeStep=0.01, IntegrationTime=0.5, Integrator=1)
    f.UpdatePipeline()
    ref = reference(flag_code=4, pt_IC_index=100, time_step=0.01, Integration_time=0.5, integrator="rk4")
    lines = fetch(f, 2)
    pts_ref, _, _, _ = ref["data"]["trajectory"].as_arrays()
    check(lines.GetNumberOfLines() == 1 and lines.GetNumberOfPoints() == len(pts_ref), f"trajectory with {lines.GetNumberOfPoints()} points (reference {len(pts_ref)})")
    close("trajectory points", vnp.vtk_to_numpy(lines.GetPoints().GetData()), pts_ref, 1e-9)
    f.UpdatePipeline(8.0)
    p3 = fetch(f, 3)
    check(p3.GetNumberOfPoints() == 1, f"tracer position at t = 8.0 (input time axis): {p3.GetNumberOfPoints()} point")

    # ---- 6. tagged tracers from the Seeds input ------------------------------------------
    print("\n=== tagged tracers (flag 10) ===")
    tags = XMLPolyDataReader(FileName=TAGS)
    f = make(Analysis=10, TimeStep=0.01, IntegrationTime=0.3, NumberOfSnapshots=3)
    f.Seeds = tags
    f.UpdatePipeline()
    ref = reference(flag_code=10, infile_tag=TAGS, time_step=0.01, Integration_time=0.3, number_output_files=3)
    f.UpdatePipeline(6.0)
    tr = fetch(f, 3)
    rec = ref["data"]["snapshots"][-1]
    check(tr.GetNumberOfPoints() == len(rec["ids"]), f"{tr.GetNumberOfPoints()} tagged tracers at the end (reference {len(rec['ids'])})")
    if tr.GetNumberOfPoints() == len(rec["ids"]):
        close("tags", arr(tr, "tracer_tag"), rec["tags"], 0)
        close("RT of the tagged tracers", arr(tr, "RT"), rec["RT"], 1e-12)

    # ---- 7. steady data set, staggered release, files written ---------------------------------
    print("\n=== steady patch: staggered release (flag 5) with files ===")
    steady = LegacyVTKReader(FileNames=[PATCH])
    outdir = tempfile.mkdtemp(prefix="wsslcs_pv_")
    f = Filter(Input=steady)
    f.WSSVectors = ["POINTS", "wss"]
    f.WSSScale = 0.03
    f.Analysis = 5
    f.NumberOfReleases = 2
    f.TimeBetweenReleases = 0.1
    f.TimeStep = 0.01
    f.IntegrationTime = 0.3
    f.NumberOfSnapshots = 3
    f.OutputDirectory = outdir
    f.OutputPrefix = "P3"
    f.UpdatePipeline()
    ref = reference(flag_code=5, Steady_flag=1, infile=PATCH, Num_stag=2, stag_delta=0.1, time_step=0.01, Integration_time=0.3, number_output_files=3)
    surf = fetch(f, 0)
    close("RT (staggered)", arr(surf, "RT"), np.nan_to_num(ref["data"]["RT_vertex"]), 1e-12)
    close("WSSET (staggered)", arr(surf, "WSSET", cell=True), ref["data"]["WSSET"], 1e-12)
    written = sorted(os.listdir(outdir))
    check(any(n.startswith("P3_ETstag") for n in written) and any(n.endswith(".pvd") for n in written), f"files written: {written}")
    shutil.rmtree(outdir, ignore_errors=True)

    print("\n%s" % (f"{failures} FAILURES" if failures else "ALL PLUGIN TESTS PASSED"))
    return failures


if __name__ == "__main__":
    try:
        sys.exit(1 if main() else 0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
