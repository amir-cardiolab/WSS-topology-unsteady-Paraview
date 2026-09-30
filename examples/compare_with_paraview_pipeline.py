"""Compares the TSVI of the plugin with the equivalent ParaView pipeline built from
standard filters:  Calculator norm(WSS) -> Gradient -> Calculator trace
(Gradient_0 + Gradient_4 + Gradient_8) -> Temporal Statistics (stddev).

    pvpython examples/compare_with_paraview_pipeline.py [frames...]

Without arguments the synthetic sequence of examples/data is used.  Temporal
Statistics' stddev is the population RMS deviation, i.e. the (1/T) integral of the
TSVI definition; the nodal TSVI_point of the plugin (divergence averaged to the nodes at
every frame, then the RMS over the cycle) agrees with it to a fraction of a percent, the
per-triangle TSVI is sharper (larger near the fixed points)."""
import os
import sys

import numpy as np
from paraview.simple import Calculator, Gradient, LegacyVTKReader, LoadPlugin, TemporalStatistics
from vtkmodules.util import numpy_support as vnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
frames = sys.argv[1:] or [os.path.join(HERE, "data", f"Carotid_patch_synthetic_unsteady-{k}.vtk") for k in range(8)]
LoadPlugin(os.path.join(ROOT, "WSSLCSPlugin.py"), remote=True, ns=globals())


def out(proxy, port=0):
    return proxy.GetClientSideObject().GetOutputDataObject(port)


def arr(pd, name, cell=False):
    a = (pd.GetCellData() if cell else pd.GetPointData()).GetArray(name)
    return None if a is None else vnp.vtk_to_numpy(a).astype(float)


reader = LegacyVTKReader(FileNames=frames)
reader.UpdatePipelineInformation()
vec = "wss"
# ---- standard-filter pipeline -------------------------------------------------------------
calc1 = Calculator(Input=reader, Function=f"norm({vec})", ResultArrayName="W")
grad = Gradient(Input=calc1, ScalarArray=["POINTS", "W"], ResultArrayName="Gradient")
calc2 = Calculator(Input=grad, Function="Gradient_0+Gradient_4+Gradient_8", ResultArrayName="DIVW")
ts = TemporalStatistics(Input=calc2)
ts.UpdatePipeline()
pv_tsvi = arr(out(ts), "DIVW_stddev")
pv_mean = arr(out(ts), "DIVW_average")
# ---- the plugin ---------------------------------------------------------------------------
f = WSSSurfaceTransport(Input=reader)                  # noqa: F821
f.WSSVectors = ["POINTS", vec]
f.Analysis = "Time-series WSS metrics: TAWSS, OSI, RRT, WSS divergence, normalized WSS divergence, TSVI (unsteady data)"
f.UpdatePipeline()
surf = out(f, 0)
ours_pt, ours_mean, valid_pt = arr(surf, "TSVI_point"), arr(surf, "NormWSSdiv_TA_point"), arr(surf, "TSVI_valid_point")
ok = np.isfinite(pv_tsvi) & (valid_pt > 0)
d = ours_pt[ok] - pv_tsvi[ok]
den = np.maximum(pv_tsvi[ok], 1e-12)
print(f"points compared: {np.count_nonzero(ok)} of {len(ok)} (NaN in the ParaView result or zero-WSS nodes excluded)")
print(f"TSVI  ParaView pipeline: mean {np.mean(pv_tsvi[ok]):.5f}  max {np.max(pv_tsvi[ok]):.4f}")
print(f"TSVI  plugin TSVI_point: mean {np.mean(ours_pt[ok]):.5f}  max {np.max(ours_pt[ok]):.4f}")
print(f"median |difference| / ParaView value: {np.median(np.abs(d) / den):.4f}; correlation {np.corrcoef(ours_pt[ok], pv_tsvi[ok])[0, 1]:.5f}")
dm = ours_mean[ok] - pv_mean[ok]
print(f"time-averaged normalized divergence: median |difference| {np.median(np.abs(dm)):.5f} (median |value| {np.median(np.abs(pv_mean[ok])):.4f}), "
      f"correlation {np.corrcoef(ours_mean[ok], pv_mean[ok])[0, 1]:.5f}")
cell = arr(surf, "TSVI", cell=True)
print(f"per-triangle TSVI (cell array, no nodal averaging): mean {np.mean(cell[arr(surf, 'TSVI_valid', cell=True) > 0]):.5f}  max {np.max(cell):.4f}")
