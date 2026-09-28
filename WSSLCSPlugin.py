r"""
WSSLCSPlugin.py — ParaView plugin: WSS surface transport (WSSLCS).

Adds the filter "WSS Surface Transport (WSSLCS)" (menu Filters > WSSLCS) which runs
every analysis of the WSSLCS software on a wall shear stress (WSS) surface loaded in
ParaView, steady (one time step) or unsteady (a file series = the frames of one
period):

  * surface tracers, residence time and WSS exposure time (WSSET), staggered release,
    wall-normal diffusion / divergence models, flux from the wall, tagged tracers
    (seeds from a second input), single tracers;
  * WSS fixed points and their stable / unstable manifolds (WSS LCS), of the current
    time step or of the time averaged field;
  * fixed points tracked in time and their exposure time;
  * time-series WSS metrics: TAWSS, OSI, RRT, time-averaged WSS divergence, TSVI.

Outputs (four ports): 0 "Surface" (the input surface with the result arrays),
1 "Fixed points", 2 "Lines" (manifolds, tracer paths, single trajectory) and
3 "Tracers" (time dependent: tracer snapshots or fixed points in time; animate it
with ParaView's time controls).

Loading: Tools > Manage Plugins > Load New ... > this file.  The folder "wsslcs"
(the Python package) must stay next to this file (on the server for a remote
pvserver).  Batch use:  pvpython -c "from paraview.simple import *; LoadPlugin(...)".
"""
import hashlib
import os
import sys

import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smhint, smproperty, smproxy
from vtkmodules.util import numpy_support as vnp
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPointSet, vtkPolyData
from vtkmodules.vtkCommonExecutionModel import vtkExecutive
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wsslcs import api, cli, exposure, io_vtk, metrics            # noqa: E402
from wsslcs.field import FieldSequence, SurfaceVectorField          # noqa: E402
from wsslcs.fixedpoints import fixed_points_polydata                # noqa: E402
from wsslcs.manifolds import manifolds_polydata                     # noqa: E402
from wsslcs.mesh import SurfaceMesh                                 # noqa: E402
from wsslcs.params import FLAG_CODES, Parameters                    # noqa: E402
from wsslcs.tracer import RunCancelled                              # noqa: E402

__version__ = "1.0"

TRACER_CODES = (1, 2, 5, 6, 7, 9, 10)
SINGLE_CODES = (3, 4)
MODES = [
    (12, "Fixed points + stable/unstable manifolds (WSS LCS)"),
    (11, "Fixed points only"),
    (1, "Surface tracers: residence time + WSS exposure time"),
    (2, "Surface tracers only (snapshots)"),
    (5, "Staggered release: WSS exposure time"),
    (3, "Single tracer released at coordinates"),
    (4, "Single tracer released at a vertex"),
    (6, "Tracers with wall-normal diffusion (random walk)"),
    (7, "Tracers with wall-normal velocity from the WSS divergence"),
    (9, "Flux from the wall with diffusion"),
    (10, "Tagged tracers (seeds from the Seeds input)"),
    (8, "Fixed points tracked in time (unsteady data)"),
    (13, "Time-series WSS metrics: TAWSS, OSI, RRT, WSS divergence, TSVI (unsteady data)"),
]


def _print_error(msg):
    try:
        from paraview import print_error
        print_error("WSSLCS: " + msg)
    except Exception:  # noqa: BLE001
        print("WSSLCS ERROR: " + msg, file=sys.stderr)


class MemorySequence(FieldSequence):
    """FieldSequence whose frames are numpy vector arrays in memory (the time steps
    fetched through the ParaView pipeline) instead of files."""

    def __init__(self, params, mesh, frames):
        self.frames = frames
        super().__init__(params, mesh=mesh)

    def file_name(self, idx):
        return f"<time step {idx}>"

    def field_at_index(self, idx):
        idx = self.wrap(idx)
        f = self._cache.get(idx)
        if f is None:
            f = SurfaceVectorField(self.mesh, self.frames[idx], scale=self.p.WSS_SCALE, name=self.vector_name)
            self._cache[idx] = f
            self._order.append(idx)
            while len(self._order) > self.max_cache:
                self._cache.pop(self._order.pop(0), None)
        return f


# ---------------------------------------------------------------------------
# ServerManager XML (properties, groups, output ports)
# ---------------------------------------------------------------------------
def _vis(values):
    return (f'<Hints><PropertyWidgetDecorator type="GenericDecorator" mode="visibility" '
            f'property="Analysis" values="{values}" /></Hints>')


def _prop(kind, name, label, command, default, doc, extra="", nelem=1, advanced=False):
    pv = ' panel_visibility="advanced"' if advanced else ""
    return (f'<{kind} name="{name}" command="{command}" number_of_elements="{nelem}" '
            f'default_values="{default}"{pv}>\n  <Documentation>{label}. {doc}</Documentation>\n  {extra}\n</{kind}>')


def _int(name, label, command, default, doc, values=None, advanced=False):
    return _prop("IntVectorProperty", name, label, command, default, doc, _vis(values) if values else "", advanced=advanced)


def _bool(name, label, command, default, doc, values=None, advanced=False):
    extra = '<BooleanDomain name="bool" />' + (_vis(values) if values else "")
    return _prop("IntVectorProperty", name, label, command, default, doc, extra, advanced=advanced)


def _double(name, label, command, default, doc, values=None, advanced=False, nelem=1):
    return _prop("DoubleVectorProperty", name, label, command, default, doc, _vis(values) if values else "", nelem=nelem, advanced=advanced)


def _enum(name, label, command, default, doc, entries, values=None):
    ent = "".join(f'<Entry value="{v}" text="{t}" />' for v, t in entries)
    return _prop("IntVectorProperty", name, label, command, default, doc, f'<EnumerationDomain name="enum">{ent}</EnumerationDomain>' + (_vis(values) if values else ""))


TRACERS = "1 2 5 6 7 9 10"
TIMED = "1 2 3 4 5 6 7 8 9 10"

INPUTS_XML = """
<Documentation short_help="Near-wall (surface) transport driven by a WSS vector field: tracers, residence and exposure time, fixed points, manifolds, time-series metrics."
  long_help="Runs the WSSLCS analyses (Arzani et al. 2016, 2017, 2018) on a WSS surface: surface tracers with residence time and WSS exposure time, staggered release, wall-normal models, tagged tracers, single tracers, fixed points and stable/unstable manifolds (WSS LCS), fixed points tracked in time and the time-series metrics TAWSS, OSI, RRT, WSS divergence and TSVI. With a file series (the frames of one period) the filter fetches all time steps through the pipeline; longer integrations continue periodically." />
<InputProperty name="Input" command="SetInputConnection" port_index="0">
  <ProxyGroupDomain name="groups"><Group name="sources" /><Group name="filters" /></ProxyGroupDomain>
  <DataTypeDomain name="input_type"><DataType value="vtkDataSet" /></DataTypeDomain>
  <InputArrayDomain name="inputs_array" attribute_type="point" number_of_components="3" />
  <Documentation>The WSS surface: a triangulated surface (any dataset; unstructured grids are converted to their surface) with a 3-component point array holding the WSS vectors. A file series is the periodic sequence of WSS frames.</Documentation>
</InputProperty>
<InputProperty name="Seeds" command="SetInputConnection" port_index="1" optional="1">
  <ProxyGroupDomain name="groups"><Group name="sources" /><Group name="filters" /></ProxyGroupDomain>
  <DataTypeDomain name="input_type"><DataType value="vtkPointSet" /></DataTypeDomain>
  <Documentation>Optional: points used as seeds for "Tagged tracers" (an integer point array named tracer_tag gives the tags). The seeds are projected on the surface.</Documentation>
</InputProperty>
<DoubleVectorProperty information_only="1" name="TimestepValues" repeatable="1">
  <TimeStepsInformationHelper />
  <Documentation>Time values of the tracer snapshots (or of the fixed points in time).</Documentation>
</DoubleVectorProperty>
<OutputPort index="0" name="Surface" />
<OutputPort index="1" name="Fixed points" />
<OutputPort index="2" name="Lines" />
<OutputPort index="3" name="Tracers" />
"""

PROPERTIES_XML = "\n".join([
    _enum("Analysis", "Analysis", "SetAnalysis", 12, "Which analysis to run (the flag_code of the WSSLCS program).", MODES),
    """<StringVectorProperty name="SelectInputVectors" command="SetInputArrayToProcess" number_of_elements="5" element_types="0 0 0 0 2" animateable="0">
  <ArrayListDomain name="array_list" attribute_type="Vectors" input_domain_name="inputs_array">
    <RequiredProperties><Property function="Input" name="Input" /></RequiredProperties>
  </ArrayListDomain>
  <Documentation>The point vector array with the WSS.</Documentation>
</StringVectorProperty>""",
    _double("WSSScale", "WSS scale", "SetWSSScale", 0.03, "Tracer velocity = WSS scale x WSS (delta_n / mu: the near-wall velocity at the distance delta_n from the wall). Default 0.03."),
    _bool("UseAllTimeSteps", "Use all time steps (periodic sequence)", "SetUseAllTimeSteps", 1,
          "On: all time steps of the input are the frames of one period; they are fetched through the pipeline and the integration continues periodically beyond the last frame. Off: only the current time step is used (steady analysis)."),
    _double("TimeBetweenFrames", "Time between frames (0 = from the data)", "SetTimeBetweenFrames", 0.0,
            "Physical time between consecutive frames of the sequence. 0 uses the time values of the input (file series without time information count 0, 1, 2, ...)."),
    _int("ReleaseFrame", "Release at frame (index)", "SetReleaseFrame", 0, "Index of the frame at which the integration starts (0 = first time step)."),
    _double("TimeStep", "Time step", "SetTimeStep", 0.01, "Integration time step (time units of the data).", TIMED),
    _double("IntegrationTime", "Integration time", "SetIntegrationTime", 1.0, "Total integration time; longer than the period of the sequence wraps around (periodic flow).", TIMED),
    _enum("Direction", "Direction", "SetDirection", 0, "Forward or backward integration in time.", [(0, "forward"), (1, "backward")], TIMED),
    _enum("Integrator", "Integrator", "SetIntegrator", 0, "Explicit Euler (as in the papers) or RK4.", [(0, "Euler"), (1, "RK4")], "1 2 3 4 5 6 7 9 10"),
    _int("NumberOfSnapshots", "Snapshots", "SetNumberOfSnapshots", 10, "Number of snapshots of the tracer positions (or of the fixed points in time) provided on the time dependent output port 'Tracers'.", "1 2 5 6 7 8 9 10"),
    _bool("ReleaseAtCentroids", "Also release at triangle centroids", "SetReleaseAtCentroids", 0, "Release tracers at the triangle centroids in addition to the vertices.", "1 2 5 6 7 9"),
    _bool("TrajectoryLines", "Tracer paths (polylines on 'Lines')", "SetTrajectoryLines", 0, "Record the tracer positions at the snapshot times and output the paths as polylines.", TRACERS),
    _int("NumberOfReleases", "Number of releases", "SetNumberOfReleases", 1, "Staggered release: number of releases of all tracers.", "5"),
    _double("TimeBetweenReleases", "Time between releases", "SetTimeBetweenReleases", 0.03, "Staggered release: time between two releases.", "5"),
    _double("DiffusionCoefficient", "Diffusion coefficient", "SetDiffusionCoefficient", 5e-5, "Wall-normal random walk: y += N(0, sqrt(2 D dt)).", "6 9"),
    _double("MaxWallNormalDistance", "Max. wall-normal distance", "SetMaxWallNormalDistance", 0.06, "Tracers farther from the wall leave the near-wall region.", "6 7 9"),
    _double("Viscosity", "Viscosity mu", "SetViscosity", 0.04, "delta_n = WSS scale x mu is the initial distance of the tracers from the wall.", "6 7 9"),
    _int("RandomSeed", "Random seed (0 = random)", "SetRandomSeed", 0, "Seed of the random walk.", "6 9"),
    _double("ReleasePoint", "Release point (x, y, z)", "SetReleasePoint", "0 0 0", "Coordinates of the single tracer release point (snapped to the closest surface point).", "3", nelem=3),
    _int("ReleaseVertex", "Release vertex index", "SetReleaseVertex", 0, "Vertex at which the single tracer is released.", "4"),
    _bool("IgnoreBoundaryTriangles", "Ignore boundary triangles", "SetIgnoreBoundaryTriangles", 1, "Do not report fixed points in triangles touching the boundary of the surface.", "8 11 12"),
    _double("ZeroTolerance", "Zero vector tolerance", "SetZeroTolerance", 1e-10, "Vectors shorter than this are treated as zero when locating fixed points.", "8 11 12"),
    _double("ManifoldStep", "Manifold step / sqrt(area)", "SetManifoldStep", 0.2, "Arc-length step of the manifold integration relative to the local triangle size.", "12"),
    _int("ManifoldMaxSteps", "Manifold max. steps", "SetManifoldMaxSteps", 20000, "Maximum number of steps per manifold branch.", "12"),
    _double("ManifoldMaxLength", "Manifold max. length (0 = auto)", "SetManifoldMaxLength", 0.0, "Maximum length of a manifold branch (0: 25 times the bounding radius).", "12"),
    _double("ManifoldPerturbation", "Perturbation / sqrt(area)", "SetManifoldPerturbation", 0.1, "Distance from the saddle at which the manifold integration starts.", "12"),
    _double("CaptureRadius", "Capture radius / sqrt(area)", "SetCaptureRadius", 0.5, "A manifold stops when it comes this close to another fixed point.", "12"),
    _int("MaxCrossings", "Max. triangle crossings per step", "SetMaxCrossings", 50, "Maximum number of triangle crossings of a tracer within one time step.", advanced=True),
    """<StringVectorProperty name="OutputDirectory" command="SetOutputDirectory" number_of_elements="1" default_values="" panel_visibility="advanced">
  <FileListDomain name="files" />
  <Hints><UseDirectoryName /></Hints>
  <Documentation>Write VTK files to folder (optional): also write the result files of the command-line program (RT, ET, tracer snapshots + .pvd, fixed points, manifolds, SingET, WSSmetrics, ...) into this folder.</Documentation>
</StringVectorProperty>""",
    _prop("StringVectorProperty", "OutputPrefix", "File prefix", "SetOutputPrefix", "wsslcs", "Prefix of the files written to the folder above.", advanced=True),
    """<PropertyGroup label="Data"><Property name="SelectInputVectors" /><Property name="WSSScale" /><Property name="UseAllTimeSteps" /><Property name="TimeBetweenFrames" /><Property name="ReleaseFrame" /></PropertyGroup>
<PropertyGroup label="Integration"><Property name="TimeStep" /><Property name="IntegrationTime" /><Property name="Direction" /><Property name="Integrator" /><Property name="NumberOfSnapshots" /><Property name="ReleaseAtCentroids" /><Property name="TrajectoryLines" /><Property name="NumberOfReleases" /><Property name="TimeBetweenReleases" /><Property name="DiffusionCoefficient" /><Property name="MaxWallNormalDistance" /><Property name="Viscosity" /><Property name="RandomSeed" /><Property name="ReleasePoint" /><Property name="ReleaseVertex" /><Property name="MaxCrossings" /></PropertyGroup>
<PropertyGroup label="Fixed points and manifolds"><Property name="IgnoreBoundaryTriangles" /><Property name="ZeroTolerance" /><Property name="ManifoldStep" /><Property name="ManifoldMaxSteps" /><Property name="ManifoldMaxLength" /><Property name="ManifoldPerturbation" /><Property name="CaptureRadius" /></PropertyGroup>
<PropertyGroup label="Files"><Property name="OutputDirectory" /><Property name="OutputPrefix" /></PropertyGroup>""",
])


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------
@smproxy.filter(name="WSSLCSSurfaceTransport", label="WSS Surface Transport")
@smhint.xml('<ShowInMenu category="WSSLCS" />')
@smproperty.xml(PROPERTIES_XML)
@smproperty.xml(INPUTS_XML)
class WSSLCSSurfaceTransport(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=4, outputType="vtkPolyData")
        self._code = 12
        self._scale = 0.03
        self._use_all = True
        self._dt_override = 0.0
        self._release = 0
        self._h = 0.01
        self._T = 1.0
        self._direction = 0
        self._integrator = 0
        self._nout = 10
        self._highres = 0
        self._lines = 0
        self._num_stag = 1
        self._stag_delta = 0.03
        self._diff = 5e-5
        self._max_yn = 0.06
        self._mu = 0.04
        self._seed = 0
        self._pt = [0.0, 0.0, 0.0]
        self._vertex = 0
        self._exclude_boundary = 1
        self._zero_tol = 1e-10
        self._m_step = 0.2
        self._m_max_steps = 20000
        self._m_max_length = 0.0
        self._m_pert = 0.1
        self._capture = 0.5
        self._max_cross = 50
        self._outdir = ""
        self._prefix = "wsslcs"
        # pipeline state
        self._in_times = []
        self._pending = None          # frame indices still to fetch (multi-pass gathering)
        self._frames = []
        self._cache = None

    # ---- property setters ----------------------------------------------------------
    def _set(self, attr, value):
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            self.Modified()

    def SetAnalysis(self, v): self._set("_code", int(v))
    def SetWSSScale(self, v): self._set("_scale", float(v))
    def SetUseAllTimeSteps(self, v): self._set("_use_all", bool(v))
    def SetTimeBetweenFrames(self, v): self._set("_dt_override", float(v))
    def SetReleaseFrame(self, v): self._set("_release", int(v))
    def SetTimeStep(self, v): self._set("_h", float(v))
    def SetIntegrationTime(self, v): self._set("_T", float(v))
    def SetDirection(self, v): self._set("_direction", int(v))
    def SetIntegrator(self, v): self._set("_integrator", int(v))
    def SetNumberOfSnapshots(self, v): self._set("_nout", int(v))
    def SetReleaseAtCentroids(self, v): self._set("_highres", int(v))
    def SetTrajectoryLines(self, v): self._set("_lines", int(v))
    def SetNumberOfReleases(self, v): self._set("_num_stag", int(v))
    def SetTimeBetweenReleases(self, v): self._set("_stag_delta", float(v))
    def SetDiffusionCoefficient(self, v): self._set("_diff", float(v))
    def SetMaxWallNormalDistance(self, v): self._set("_max_yn", float(v))
    def SetViscosity(self, v): self._set("_mu", float(v))
    def SetRandomSeed(self, v): self._set("_seed", int(v))
    def SetReleasePoint(self, x, y, z): self._set("_pt", [float(x), float(y), float(z)])
    def SetReleaseVertex(self, v): self._set("_vertex", int(v))
    def SetIgnoreBoundaryTriangles(self, v): self._set("_exclude_boundary", int(v))
    def SetZeroTolerance(self, v): self._set("_zero_tol", float(v))
    def SetManifoldStep(self, v): self._set("_m_step", float(v))
    def SetManifoldMaxSteps(self, v): self._set("_m_max_steps", int(v))
    def SetManifoldMaxLength(self, v): self._set("_m_max_length", float(v))
    def SetManifoldPerturbation(self, v): self._set("_m_pert", float(v))
    def SetCaptureRadius(self, v): self._set("_capture", float(v))
    def SetMaxCrossings(self, v): self._set("_max_cross", int(v))
    def SetOutputDirectory(self, v): self._set("_outdir", str(v or ""))
    def SetOutputPrefix(self, v): self._set("_prefix", str(v or "wsslcs"))

    # ---- helpers -------------------------------------------------------------------
    def _unsteady(self):
        return self._use_all and len(self._in_times) > 1

    def _frame_dt(self):
        if self._dt_override > 0:
            return self._dt_override
        if len(self._in_times) > 1:
            return float(self._in_times[1] - self._in_times[0]) or 1.0
        return 1.0

    def _release_time(self):
        if self._unsteady():
            k = min(max(self._release, 0), len(self._in_times) - 1)
            return float(self._in_times[k])
        return 0.0

    def _time_factor(self):
        """Conversion from the elapsed physical time to the time axis of the input
        (a file series without time information counts 0, 1, 2, ... while the physical
        spacing is 'Time between frames'), so that the tracer snapshots animate in step
        with the WSS frames."""
        if self._unsteady() and self._dt_override > 0:
            din = float(self._in_times[1] - self._in_times[0])
            return din / self._dt_override if din > 0 else 1.0
        return 1.0

    def _n_steps(self):
        return max(int(np.floor(self._T / self._h + 1e-9)), 1) if self._h > 0 else 1

    def _snapshot_times(self):
        """ParaView time values of the time dependent output (port 3)."""
        code = self._code
        if code not in TRACER_CODES and code not in SINGLE_CODES and code != 8:
            return []
        n = self._n_steps()
        h = self._h
        if code in SINGLE_CODES:
            stride = max(int(np.ceil((n + 1) / 1000.0)), 1)
            elapsed = [i * h for i in range(0, n + 1, stride)]
            if elapsed[-1] < n * h:
                elapsed.append(n * h)
        else:
            out_freq = max(n // max(self._nout, 1), 1)
            elapsed = [i * h for i in range(n) if i == 0 or i % out_freq == 0]
            if code != 8:
                elapsed.append(n * h)
        t0, sgn, fac = self._release_time(), (-1.0 if self._direction == 1 else 1.0), self._time_factor()
        return sorted(set(round(t0 + sgn * e * fac, 10) for e in elapsed))

    def _params_key(self):
        return (self._code, self._scale, self._use_all, self._dt_override, self._release, self._h, self._T, self._direction,
                self._integrator, self._nout, self._highres, self._lines, self._num_stag, self._stag_delta, self._diff,
                self._max_yn, self._mu, self._seed, tuple(self._pt), self._vertex, self._exclude_boundary, self._zero_tol,
                self._m_step, self._m_max_steps, self._m_max_length, self._m_pert, self._capture, self._max_cross,
                self._outdir, self._prefix)

    def _parameters(self, steady, n_frames, dt):
        p = Parameters(infile="paraview-input", vector_array=self._array_name or "wss", output_prefix=self._prefix,
                       output_dir=self._outdir or ".", output_format="vtp",
                       FILE_index_start=0, FILE_index_end=max(n_frames - 1, 0), index_delta=1, delta_t_file=dt,
                       index_first=0 if steady else min(max(self._release, 0), n_frames - 1),
                       Steady_flag=1 if steady else 0, flag_code=self._code, WSS_SCALE=self._scale,
                       time_step=self._h, Integration_time=self._T, flag_int_direction=self._direction,
                       integrator="rk4" if self._integrator == 1 else "euler", number_output_files=self._nout,
                       HighRes_flag=self._highres, Num_stag=self._num_stag, stag_delta=self._stag_delta,
                       Diff_coef=self._diff, max_yn=self._max_yn, mu=self._mu, random_seed=self._seed,
                       pt_IC=list(self._pt), pt_IC_index=self._vertex,
                       exclude_boundary_fixed_points=self._exclude_boundary, zero_vector_tolerance=self._zero_tol,
                       manifold_step_fraction=self._m_step, manifold_max_steps=self._m_max_steps,
                       manifold_max_length=self._m_max_length, manifold_perturbation=self._m_pert,
                       fixed_point_capture_radius=self._capture, write_trajectory_lines=self._lines,
                       max_crossings_per_step=self._max_cross, verbose=0)
        p.validate()
        return p

    # ---- pipeline --------------------------------------------------------------------
    def FillInputPortInformation(self, port, info):
        info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet" if port == 0 else "vtkPointSet")
        if port == 1:
            info.Set(self.INPUT_IS_OPTIONAL(), 1)
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        info_in = inInfo[0].GetInformationObject(0)
        self._in_times = list(info_in.Get(SDDP.TIME_STEPS())) if info_in.Has(SDDP.TIME_STEPS()) else []
        # the snapshot times are advertised on every port (ParaView's time keeper reads
        # port 0); the static outputs simply do not change with time
        times = self._snapshot_times()
        for port in range(4):
            oi = outInfo.GetInformationObject(port)
            oi.Remove(SDDP.TIME_STEPS())
            oi.Remove(SDDP.TIME_RANGE())
            if times:
                oi.Set(SDDP.TIME_STEPS(), times, len(times))
                oi.Set(SDDP.TIME_RANGE(), [times[0], times[-1]], 2)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        info_in = inInfo[0].GetInformationObject(0)
        if self._unsteady():
            if self._pending is None and (self._cache is None or self._cache["params_key"] != self._params_key()
                                          or self._cache["in_times"] != tuple(self._in_times)):
                self._frames = [None] * len(self._in_times)
                self._pending = list(range(len(self._in_times)))
            if self._pending:
                info_in.Set(SDDP.UPDATE_TIME_STEP(), float(self._in_times[self._pending[0]]))
        return 1

    def _frame_index(self, inp):
        """Index of the input time step currently held by the input data object."""
        di = inp.GetInformation()
        if di.Has(vtkDataObject.DATA_TIME_STEP()) and self._in_times:
            t = di.Get(vtkDataObject.DATA_TIME_STEP())
            return int(np.argmin(np.abs(np.asarray(self._in_times, dtype=float) - t)))
        return self._pending[0] if self._pending else 0

    def _read_input(self, inp):
        pd = io_vtk.as_triangle_polydata(inp)
        info = self.GetInputArrayInformation(0)
        name = info.Get(vtkDataObject.FIELD_NAME()) if info is not None and info.Has(vtkDataObject.FIELD_NAME()) else None
        arr = pd.GetPointData().GetArray(name) if name else None
        if arr is None:
            for i in range(pd.GetPointData().GetNumberOfArrays()):
                cand = pd.GetPointData().GetArray(i)
                if cand is not None and cand.GetNumberOfComponents() == 3:
                    arr, name = cand, cand.GetName()
                    break
        if arr is None:
            raise ValueError("the input has no 3-component point array (WSS vectors)")
        self._array_name = name
        vec = vnp.vtk_to_numpy(arr).astype(np.float64).reshape(-1, 3)
        return pd, vec

    def RequestData(self, request, inInfo, outInfo):
        try:
            return self._request_data(request, inInfo, outInfo)
        except RunCancelled:
            self._pending = None
            _print_error("cancelled")
            return 0
        except Exception as exc:  # noqa: BLE001
            self._pending = None
            import traceback
            _print_error(f"{exc}\n{traceback.format_exc()}")
            return 0

    def _request_data(self, request, inInfo, outInfo):
        inp = vtkDataSet.GetData(inInfo[0])
        if inp is None or inp.GetNumberOfPoints() == 0:
            raise ValueError("empty input")
        pd, vec = self._read_input(inp)
        pts = io_vtk.polydata_points(pd)
        tris = io_vtk.polydata_triangles(pd)
        geom_key = hashlib.blake2b(pts.tobytes() + tris.tobytes(), digest_size=16).hexdigest()
        unsteady = self._unsteady()
        if unsteady:
            if self._pending is None:
                if self._cache is None or self._cache["params_key"] != self._params_key() \
                        or self._cache["in_times"] != tuple(self._in_times) or self._cache["geom_key"] != geom_key:
                    self._frames = [None] * len(self._in_times)
                    self._pending = list(range(len(self._in_times)))
            if self._pending is not None:
                k = self._frame_index(inp)
                self._frames[k] = vec
                if k in self._pending:
                    self._pending.remove(k)
                if self._pending:
                    request.Set(SDDP.CONTINUE_EXECUTING(), 1)
                    self.UpdateProgress(0.05 * (1 - len(self._pending) / max(len(self._in_times), 1)))
                    return 1
                request.Remove(SDDP.CONTINUE_EXECUTING())
                self._pending = None
                if any(f is None for f in self._frames):
                    raise ValueError("could not fetch all time steps of the input")
                self._run(pd, pts, tris, list(self._frames), geom_key, steady=False)
        else:
            vec_key = hashlib.blake2b(vec.tobytes(), digest_size=16).hexdigest()
            if self._cache is None or self._cache["params_key"] != self._params_key() or self._cache["geom_key"] != geom_key \
                    or self._cache.get("vec_key") != vec_key or self._cache["in_times"] != tuple(self._in_times):
                self._run(pd, pts, tris, [vec], geom_key, steady=True, vec_key=vec_key)
        self._produce(request, outInfo)
        return 1

    # ---- the analysis --------------------------------------------------------------------
    def _run(self, pd, pts, tris, frames, geom_key, steady, vec_key=None):
        code = self._code
        params = self._parameters(steady, len(frames), self._frame_dt())
        write_files = bool(self._outdir)
        if write_files:
            os.makedirs(self._outdir, exist_ok=True)
        if code == 10:
            seeds = vtkPointSet.GetData(self.GetInputInformation(1, 0)) if self.GetNumberOfInputConnections(1) > 0 else None
            if seeds is None or seeds.GetNumberOfPoints() == 0:
                raise ValueError("'Tagged tracers' needs seed points on the 'Seeds' input (point array tracer_tag)")
            params.seed_points = io_vtk.polydata_points(seeds)
            tag_arr = seeds.GetPointData().GetArray("tracer_tag")
            if tag_arr is None:
                for i in range(seeds.GetPointData().GetNumberOfArrays()):
                    cand = seeds.GetPointData().GetArray(i)
                    if cand is not None and cand.GetNumberOfComponents() == 1:
                        tag_arr = cand
                        break
            params.seed_tags = (vnp.vtk_to_numpy(tag_arr).astype(np.int64).reshape(-1) if tag_arr is not None
                                else np.zeros(len(params.seed_points), dtype=np.int64))
        mesh = SurfaceMesh(pts, tris)
        seq = MemorySequence(params, mesh, frames)

        def progress(frac, msg):
            self.UpdateProgress(0.05 + 0.9 * float(frac))
            return not self.GetAbortExecute()

        if code == 13:
            out = metrics.run_metrics(params, mesh, seq, False, progress, write_files)
        elif code in SINGLE_CODES:
            out = cli._single_tracer(params, mesh, seq, False, progress, write_files)
        elif code in (11, 12):
            out = cli._topology(params, mesh, seq, False, progress, write_files)
        elif code == 8:
            out = exposure.run_fixed_point_tracking(params, mesh, seq, False, progress, write_files)
        else:
            out = exposure.run_advection(params, mesh, seq, False, progress, write_files)
        data = out.get("data", {})
        scale = float(params.WSS_SCALE) or 1.0
        field0 = seq.field_at_index(seq.index_first)
        # ---- port 0: the surface with the result arrays -------------------------------
        if code in (11, 12):
            res = data["result"]
            surface = api.surface_polydata(res)
        else:
            surface = mesh.to_polydata({"wss_magnitude": field0.mag / scale}, {"divergence": field0.divergence() / scale})
        if surface.GetNumberOfPoints() == pd.GetNumberOfPoints():
            surface.GetPointData().PassData(pd.GetPointData())
        if surface.GetNumberOfCells() == pd.GetNumberOfCells():
            surface.GetCellData().PassData(pd.GetCellData())
        point_arrays, cell_arrays = {}, {}
        fp_pd, lines_pd = vtkPolyData(), vtkPolyData()
        temporal = None
        if code in TRACER_CODES:
            rt = np.nan_to_num(data["RT_vertex"], nan=0.0)
            point_arrays["RT"] = rt
            status_v = np.zeros(mesh.n_points, dtype=np.int32)
            kind, index, copy = data["seed_kind"], data["seed_index"], data["seed_copy"]
            first = (kind == exposure.SEED_VERTEX) & (copy == 0)
            status_v[index[first]] = data["status"][first]
            point_arrays["status"] = status_v
            if data.get("y_normal") is not None:
                yv = np.zeros(mesh.n_points)
                yv[index[first]] = data["y_normal"][first]
                point_arrays["y_normal"] = yv
            if code != 2:
                cell_arrays.update({"ET": data["ET"], "ET_norm": data["ET"] / np.sqrt(mesh.area), "WSSET": data["WSSET"]})
                point_arrays["WSSET_point"] = mesh.cell_to_point(data["WSSET"])
                point_arrays["ET_point"] = mesh.cell_to_point(data["ET"])
            temporal = {"kind": "tracers", "records": data["snapshots"], "tags": data.get("tags")}
            if self._lines:
                lines_pd = self._trajectory_lines(data)
        elif code == 8:
            point_arrays.update({"SingET_nodal": data["SingET_nodal"], "SingET_nodal_eig": data["SingET_nodal_eig"]})
            cell_arrays.update({"SingET": data["SingET"], "SingET_norm": data["SingET"] / np.sqrt(mesh.area)})
            temporal = {"kind": "fixedpoints", "records": data["snapshots"], "scale": scale}
        elif code == 13:
            for name, (values, loc) in data["metrics"].items():
                (point_arrays if loc == "point" else cell_arrays)[name] = values
        elif code in SINGLE_CODES:
            traj = data["trajectory"]
            tp, tt, ttri, tspeed = traj.as_arrays()
            lines_pd = io_vtk.polylines_polydata([tp], {"time": [tt], "wss_magnitude": [tspeed / scale], "triangle": [ttri.astype(np.int32)]},
                                                 {"end_reason": [traj.end_reason], "length": [traj.length]})
            temporal = {"kind": "single", "points": tp, "times": tt, "speed": tspeed / scale}
        else:
            res = data["result"]
            fp_pd = fixed_points_polydata(res.fixed_points, scale)
            if code == 12:
                lines_pd = manifolds_polydata(res.branches, scale)
        io_vtk.add_arrays(surface, point_arrays, cell_arrays)
        summary = {k: v for k, v in out.items() if k not in ("data", "files")}
        self._cache = {"params_key": self._params_key(), "in_times": tuple(self._in_times), "geom_key": geom_key, "vec_key": vec_key,
                       "surface": surface, "fixed_points": fp_pd, "lines": lines_pd, "temporal": temporal, "summary": summary,
                       "t0": self._release_time(), "sign": -1.0 if self._direction == 1 else 1.0, "factor": self._time_factor(), "built": {}}
        self.UpdateProgress(1.0)

    def _trajectory_lines(self, data):
        recs = data["snapshots"]
        n = len(data["RT"])
        paths = [[] for _ in range(n)]
        for rec in recs:
            for i, p in zip(rec["ids"], rec["positions"]):
                paths[int(i)].append(p)
        ids = [i for i in range(n) if len(paths[i]) >= 2]
        if not ids:
            return vtkPolyData()
        cell_arrays = {"particle_id": np.asarray(ids, dtype=np.int32), "seed_index": data["seed_index"][ids].astype(np.int32)}
        if data.get("tags") is not None:
            cell_arrays["tracer_tag"] = data["tags"][ids].astype(np.int32)
        return io_vtk.polylines_polydata([np.asarray(paths[i]) for i in ids], None, cell_arrays)

    # ---- outputs ------------------------------------------------------------------------
    def _temporal_output(self, t_req):
        c = self._cache
        temporal = c["temporal"]
        if temporal is None:
            return vtkPolyData(), None
        t0, sgn, fac = c["t0"], c["sign"], c["factor"]
        if temporal["kind"] == "single":
            times = t0 + sgn * np.abs(temporal["times"]) * fac
            k = int(np.argmin(np.abs(times - t_req))) if t_req is not None else len(times) - 1
            pd = io_vtk.points_polydata(temporal["points"][k:k + 1], {"time": temporal["times"][k:k + 1], "wss_magnitude": temporal["speed"][k:k + 1]},
                                        {"time": np.array([times[k]])})
            return pd, float(times[k])
        recs = temporal["records"]
        times = np.array([t0 + sgn * r["time"] * fac for r in recs])
        k = int(np.argmin(np.abs(times - t_req))) if t_req is not None else len(recs) - 1
        if k not in c["built"]:
            rec = recs[k]
            if temporal["kind"] == "tracers":
                arrays = {"particle_id": rec["ids"].astype(np.int32), "RT": rec["RT"], "status": rec["status"].astype(np.int32)}
                if rec.get("tags") is not None:
                    arrays["tracer_tag"] = rec["tags"].astype(np.int32)
                if rec.get("y_normal") is not None:
                    arrays["y_normal"] = rec["y_normal"]
                pd = io_vtk.points_polydata(rec["positions"], arrays, {"time": np.array([times[k]])})
            else:
                pd = fixed_points_polydata(rec["fixed_points"], temporal["scale"])
                pd.GetFieldData().AddArray(io_vtk.numpy_to_vtk_array(np.array([times[k]]), "time"))
            c["built"][k] = pd
        return c["built"][k], float(times[k])

    def _produce(self, request, outInfo):
        c = self._cache
        for port, key in ((0, "surface"), (1, "fixed_points"), (2, "lines")):
            out = vtkPolyData.GetData(outInfo.GetInformationObject(port))
            out.ShallowCopy(c[key])
        # the requested time comes from the port that triggered the execution
        port = request.Get(vtkExecutive.FROM_OUTPUT_PORT()) if request.Has(vtkExecutive.FROM_OUTPUT_PORT()) else 0
        t_req = None
        for pp in (port, 3, 0):
            oi = outInfo.GetInformationObject(pp)
            if oi.Has(SDDP.UPDATE_TIME_STEP()):
                t_req = oi.Get(SDDP.UPDATE_TIME_STEP())
                break
        pd, t = self._temporal_output(t_req)
        out3 = vtkPolyData.GetData(outInfo.GetInformationObject(3))
        out3.ShallowCopy(pd)
        if t is not None:
            for pp in range(4):
                vtkPolyData.GetData(outInfo.GetInformationObject(pp)).GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), t)

    # ---- for scripts ----------------------------------------------------------------------
    def GetSummary(self):
        """Summary dictionary of the last run (pvpython: the VTK object of the proxy)."""
        return None if self._cache is None else dict(self._cache["summary"])
