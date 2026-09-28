"""
Input-file handling for the WSSLCS Python port.

The parameter file keeps the ``VARIABLE = VALUE`` layout of the original
``P51_WSSLCS.in`` file (lines starting with ``#`` are comments, anything after
the value on a line is ignored).  Unlike the C++ code, values are matched by
NAME rather than by position, so parameters can be listed in any order and
unused ones can be omitted (defaults are used).

The meaning of ``flag_code`` follows the original program::

     1 : residence time (RT), exposure time (WSSET) and all surface tracers
     2 : all surface tracers only
     3 : one tracer released at the coordinates pt_IC
     4 : one tracer released at the vertex pt_IC_index
     5 : exposure time with staggered (repeated) release of the tracers
     6 : RT / WSSET with wall-normal diffusion (random walk in the normal direction)
     7 : RT / WSSET with the wall-normal velocity obtained from the WSS divergence
     8 : track the fixed points of the unsteady WSS field and compute their
         exposure time
     9 : flux from the wall with diffusion (all tracers start at the wall)
    10 : like 1, but the tracers carry integer tags read from infile_tag
    11 : fixed points of the (time averaged) WSS vector field          (new)
    12 : fixed points + stable/unstable manifolds, i.e. WSS LCS         (new)
    13 : time-series WSS metrics: TAWSS, OSI, RRT, time-averaged WSS
         divergence and TSVI                                             (new)

Unsteady data (Steady_flag = 0) is assumed periodic: when the integration time
exceeds the time spanned by the files, the frame after the last file is the
first file again (forward integration) and vice versa (backward integration).
"""
from __future__ import annotations

import dataclasses
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FLAG_CODES: Dict[int, str] = {
    1: "Residence time, exposure time (WSSET) and all surface tracers",
    2: "All surface tracers only",
    3: "One tracer released at the coordinates pt_IC",
    4: "One tracer released at the vertex pt_IC_index",
    5: "Exposure time with staggered release of the tracers",
    6: "RT / WSSET with wall-normal diffusion (random walk)",
    7: "RT / WSSET with wall-normal velocity from the WSS divergence",
    8: "Track the fixed points of the unsteady WSS field and their exposure time",
    9: "Flux from the wall with diffusion (tracers start at the wall)",
    10: "Like 1 but the tracers carry integer tags read from infile_tag",
    11: "Fixed points of the (time averaged) WSS vector field",
    12: "Fixed points and stable/unstable manifolds (WSS LCS)",
    13: "Time-series WSS metrics: TAWSS, OSI, RRT, time-averaged WSS divergence, TSVI",
}

ADVECTION_CODES = (1, 2, 5, 6, 7, 9, 10)


@dataclass
class Parameters:
    """All run-time parameters (original names are kept where they exist)."""

    # ---- file handling (same as the original input file) ----------------
    infile: str = ""
    output_prefix: str = "wsslcs"
    FILE_index_start: int = 0
    FILE_index_end: int = 0
    index_delta: int = 1
    delta_t_file: float = 1.0
    index_first: int = 0
    number_output_files: int = 1
    # ---- computation ----------------------------------------------------
    time_step: float = 1.0e-3
    flag_int_direction: int = 0          # 0 forward, 1 backward
    Steady_flag: int = 1                 # 1 steady (one file), 0 unsteady
    pt_IC: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pt_IC_index: int = 0
    Integration_time: float = 1.0
    WSS_SCALE: float = 1.0
    flag_code: int = 12
    Num_stag: int = 1
    stag_delta: float = 0.0
    HighRes_flag: int = 0
    Diff_coef: float = 0.0
    max_yn: float = 1.0e30
    infile_div: str = ""
    infile_tag: str = ""
    # ---- new parameters of the Python version ---------------------------
    infile_suffix: str = ".vtk"          # appended to infile+index for unsteady data
    vector_array: str = "wss"            # name of the point-vector array in the VTK file
    output_dir: str = "."
    output_format: str = "vtp"           # vtp (XML) or vtk (legacy ASCII)
    integrator: str = "euler"            # euler or rk4 (tracers); manifolds always use rk4
    mu: float = 0.04                     # dynamic viscosity (g/cm/s), used by flag_code 6, 7, 9
    exclude_boundary_fixed_points: int = 1
    zero_vector_tolerance: float = 1.0e-10
    manifold_step_fraction: float = 0.2      # arc-length step / sqrt(local triangle area)
    manifold_max_steps: int = 20000
    manifold_max_length: float = 0.0         # 0 -> automatic (25 x bounding radius)
    manifold_perturbation: float = 0.1       # initial offset from the saddle / sqrt(triangle area)
    fixed_point_capture_radius: float = 0.5  # stop a manifold within this x sqrt(area) of a fixed point
    write_trajectory_lines: int = 0          # also write the tracers as polylines (flag 1,2,5,...)
    random_seed: int = 0
    max_crossings_per_step: int = 50
    verbose: int = 1

    # ------------------------------------------------------------------
    @property
    def steady(self) -> bool:
        return int(self.Steady_flag) == 1

    @property
    def backward(self) -> bool:
        return int(self.flag_int_direction) == 1

    @property
    def direction(self) -> float:
        return -1.0 if self.backward else 1.0

    def data_file(self, index: Optional[int] = None) -> str:
        """Name of the data file for a given time index (unsteady) or of the
        single file (steady)."""
        if self.steady and index is None:
            if os.path.exists(self.infile):
                return self.infile
            cand = self.infile + self.infile_suffix
            if os.path.exists(cand):
                return cand
            return self.infile
        if index is None:
            index = self.index_first
        return f"{self.infile}{index}{self.infile_suffix}"

    def frame_indices(self) -> List[int]:
        """All file indices of the unsteady data set (one period)."""
        if self.steady:
            return [self.index_first]
        step = abs(int(self.index_delta)) or 1
        return list(range(int(self.FILE_index_start), int(self.FILE_index_end) + 1, step))

    def output_path(self, name: str, ext: Optional[str] = None) -> str:
        ext = ext or self.output_format
        if not ext.startswith("."):
            ext = "." + ext
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{self.output_prefix}_{name}{ext}")

    def validate(self) -> None:
        if int(self.flag_code) not in FLAG_CODES:
            raise ValueError(f"flag_code = {self.flag_code} is not supported. "
                             f"Valid codes: {sorted(FLAG_CODES)}")
        if not self.infile:
            raise ValueError("'infile' must be given in the parameter file")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.output_format.lower().lstrip(".") not in ("vtp", "vtk"):
            raise ValueError("output_format must be 'vtp' or 'vtk'")
        self.output_format = self.output_format.lower().lstrip(".")
        self.integrator = self.integrator.lower()
        if self.integrator not in ("euler", "rk4"):
            raise ValueError("integrator must be 'euler' or 'rk4'")
        if not self.steady:
            if self.index_delta == 0:
                raise ValueError("index_delta must be non-zero for unsteady data")
            if self.delta_t_file <= 0:
                raise ValueError("delta_t_file must be positive for unsteady data")
            if self.FILE_index_end < self.FILE_index_start:
                raise ValueError("FILE_index_end must be >= FILE_index_start")

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "Parameters":
        params = cls()
        params.update(parse_parameter_file(path))
        params.validate()
        return params

    def update(self, values: Dict[str, Any]) -> None:
        names = {f.name.lower(): f for f in dataclasses.fields(self)}
        for key, raw in values.items():
            k = key.strip()
            m = re.match(r"^(pt_IC)\[(\d)\]$", k, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(2))
                if idx > 2:
                    warnings.warn(f"ignoring parameter {key}")
                    continue
                self.pt_IC[idx] = float(raw)
                continue
            f = names.get(k.lower())
            if f is None:
                warnings.warn(f"unknown parameter '{key}' ignored")
                continue
            setattr(self, f.name, _coerce(raw, f.type, f.name))

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        lines = [f"flag_code = {self.flag_code}  ({FLAG_CODES[int(self.flag_code)]})"]
        for f in dataclasses.fields(self):
            if f.name == "flag_code":
                continue
            lines.append(f"{f.name} = {getattr(self, f.name)}")
        return "\n".join(lines)


def _coerce(raw: Any, ftype: Any, name: str) -> Any:
    if isinstance(raw, (list, tuple)):
        return list(raw)
    text = str(raw).strip()
    tname = str(ftype)
    if "int" in tname and "List" not in tname:
        try:
            return int(text)
        except ValueError:
            return int(float(text))
    if "float" in tname:
        return float(text)
    if "List" in tname:
        return [float(v) for v in re.split(r"[,\s]+", text.strip("[]() ")) if v]
    return text.strip("\"'")


def parse_parameter_file(path: str) -> Dict[str, str]:
    """Parse ``NAME = VALUE`` lines; ``#`` starts a comment.  Only the first
    token after ``=`` is used (as in the original code) unless the value is
    quoted."""
    values: Dict[str, str] = {}
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not value:
                continue
            if value[0] in "\"'":
                q = value[0]
                end = value.find(q, 1)
                value = value[1:end] if end > 0 else value[1:]
            else:
                value = value.split()[0]
            values[name] = value
    return values


TEMPLATE = """\
################################ WSSLCS INPUT FILE #####################################
# Python/VTK version.  Format: VARIABLE = VALUE  (comments start with #).
# Parameters are matched by name; unused parameters may be omitted.
###################################################################################

# infile: for steady data the full file name (.vtk/.vtp/.vtu/.ply/.stl).
#         for unsteady data the prefix; the code reads infile<index><infile_suffix>
infile = P3_TAWSS-1.vtk
infile_suffix = .vtk

# name of the point vector array (WSS) in the file (auto-detected if not found)
vector_array = wss

# outputs are written as <output_dir>/<output_prefix>_<Name>.<output_format>
output_prefix = P3
output_dir = results
output_format = vtp            # vtp (XML, binary) or vtk (legacy ASCII)

# unsteady data: first/last available file index, index spacing and time spacing.
# The files are assumed to cover one period: an integration time longer than the
# period continues with the first file after the last one (periodic flow).
FILE_index_start = 0
FILE_index_end = 0
index_delta = 1
delta_t_file = 0.015

# index of the file at which the tracers are released (unsteady only)
index_first = 0

# number of tracer snapshots written during the integration
number_output_files = 4

# time step of the tracer integration
time_step = 0.001

# 0: forward in time (unstable manifold / attracting LCS)
# 1: backward in time (stable manifold / repelling LCS)
flag_int_direction = 0

# 1: steady (single file)   0: unsteady (sequence of files)
Steady_flag = 1

# initial position of the single tracer (flag_code 3) - snapped to the surface
pt_IC[0] = 0.0
pt_IC[1] = 0.0
pt_IC[2] = 0.0

# vertex index of the single tracer (flag_code 4)
pt_IC_index = 100

# total integration time of the tracers
Integration_time = 1.0

# WSS_SCALE scales the WSS vectors (velocity = WSS_SCALE * wss). With a
# dynamic viscosity mu, WSS_SCALE = delta_n / mu gives the near-wall velocity
# at the normal distance delta_n.
WSS_SCALE = 0.0354

# flag_code: what the code computes
#  1 : residence time, exposure time (WSSET) and all surface tracers
#  2 : all surface tracers only
#  3 : one tracer from the coordinates pt_IC
#  4 : one tracer from the vertex pt_IC_index
#  5 : exposure time with staggered release
#  6 : diffusion (random walk in the wall-normal direction) added to 1
#  7 : wall-normal velocity from the WSS divergence added to 1
#  8 : track the fixed points in time and compute their exposure time (unsteady)
#  9 : flux from the wall with diffusion (all tracers start at the wall)
# 10 : like 1 but tracers carry integer tags read from infile_tag
# 11 : fixed points of the (time averaged) WSS field
# 12 : fixed points and stable/unstable manifolds (WSS LCS)
# 13 : time-series WSS metrics over one period (TAWSS, OSI, RRT, WSSdiv, TSVI)
flag_code = 12

# staggered release (flag_code 5): number of releases and time between releases
Num_stag = 1
stag_delta = 0.03

# 1: also release tracers from the element centres
HighRes_flag = 0

# diffusion coefficient (flag_code 6, 9) and maximum wall-normal distance (6, 7, 9)
Diff_coef = 5e-5
max_yn = 0.06
mu = 0.04

# optional: prefix of files with a point scalar 'wss_div' (flag_code 7). If empty,
# the surface divergence of the WSS field is computed by the code.
infile_div =

# polydata with the tracer seeds and an integer point array 'tracer_tag' (flag_code 10)
infile_tag =

# integrator for the tracers: euler (as the original code) or rk4
integrator = euler

# fixed point / manifold settings (flag_code 8, 11, 12)
exclude_boundary_fixed_points = 1
zero_vector_tolerance = 1e-10
manifold_step_fraction = 0.2
manifold_max_steps = 20000
manifold_max_length = 0
manifold_perturbation = 0.1
fixed_point_capture_radius = 0.5

# also write the tracer paths as polylines (can be large)
write_trajectory_lines = 0
random_seed = 0
verbose = 1
"""


def write_template(path: str) -> None:
    with open(path, "w") as fh:
        fh.write(TEMPLATE)


def _to_text(params: "Parameters") -> str:
    """The parameters in the layout of the template (comments preserved)."""
    lines = []
    seen = set()
    for raw in TEMPLATE.splitlines():
        code = raw.split("#", 1)[0]
        if "=" in code:
            name = code.split("=", 1)[0].strip()
            m = re.match(r"^pt_IC\[(\d)\]$", name)
            if m:
                val = params.pt_IC[int(m.group(1))]
            elif hasattr(params, name):
                val = getattr(params, name)
            else:
                lines.append(raw)
                continue
            seen.add(name)
            if isinstance(val, str) and (" " in val or "#" in val):
                val = f'"{val}"'
            comment = ""
            if "#" in raw and raw.index("#") > raw.index("="):
                comment = "   #" + raw.split("#", 1)[1]
            lines.append(f"{name} = {val}{comment}")
        else:
            lines.append(raw)
    extra = [f.name for f in dataclasses.fields(params) if f.name not in seen and f.name != "pt_IC"]
    if extra:
        lines.append("")
        for name in extra:
            lines.append(f"{name} = {getattr(params, name)}")
    return "\n".join(lines) + "\n"


Parameters.to_text = _to_text
Parameters.write = lambda self, path: open(path, "w").write(self.to_text())
