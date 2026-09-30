"""Batch example: WSS surface transport from a ParaView Python script.

    pvpython examples/pvpython_example.py [output_folder]

Loads the plugin and the synthetic 8-frame WSS sequence of examples/data, runs
(1) the residence time / WSS exposure time analysis over 1.25 periods and
(2) the fixed points + manifolds of the time averaged field, and saves the outputs
as VTK files (the tracer snapshots as a .pvd time series).
"""
import os
import sys

from paraview.simple import LegacyVTKReader, LoadPlugin, SaveData

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

LoadPlugin(os.path.join(ROOT, "WSSLCSPlugin.py"), remote=True, ns=globals())

frames = [os.path.join(HERE, "data", f"Carotid_patch_synthetic_unsteady-{k}.vtk") for k in range(8)]
reader = LegacyVTKReader(FileNames=frames)          # 8 time steps = the frames of one period

# ---- surface tracers, residence time and WSS exposure time -------------------------------
tracers = WSSSurfaceTransport(Input=reader)          # noqa: F821  (created by LoadPlugin)
tracers.WSSVectors = ["POINTS", "wss"]
tracers.WSSScale = 0.03
tracers.TimeBetweenFrames = 0.1                      # period = 8 x 0.1
tracers.ReleaseFrame = 0
tracers.Analysis = "Surface tracers: residence time + WSS exposure time"
tracers.TimeStep = 0.01
tracers.IntegrationTime = 1.0                        # 1.25 periods: wraps around
tracers.NumberOfSnapshots = 10
tracers.UpdatePipeline()
SaveData(os.path.join(OUT, "surface_RT_WSSET.vtp"), proxy=tracers, PointDataArrays=["RT", "WSSET_point"], CellDataArrays=["ET", "WSSET"])
# the time dependent output port 3 ("Tracers") saved as a time series (.pvd + one file per snapshot)
from paraview.simple import OutputPort   # noqa: E402
SaveData(os.path.join(OUT, "tracers.pvd"), proxy=OutputPort(tracers, 3), WriteTimeSteps=1)
summary = tracers.GetClientSideObject().GetSummary()
print("tracers:", {k: summary[k] for k in ("n_tracers", "n_steps", "integration_time", "RT_mean", "ET_total")})

# ---- fixed points and manifolds of the time averaged WSS field ----------------------------
topo = WSSSurfaceTransport(Input=reader)             # noqa: F821
topo.WSSVectors = ["POINTS", "wss"]
topo.WSSScale = 0.03
topo.TimeBetweenFrames = 0.1
topo.Analysis = "Fixed points + stable/unstable manifolds (WSS LCS)"
topo.UpdatePipeline()
SaveData(os.path.join(OUT, "fixed_points.vtp"), proxy=OutputPort(topo, 1))
SaveData(os.path.join(OUT, "manifolds.vtp"), proxy=OutputPort(topo, 2))
print("topology:", topo.GetClientSideObject().GetSummary())
print("outputs written to", OUT)
