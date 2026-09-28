"""
WSSLCS - Lagrangian analysis of wall shear stress (WSS) vector fields on
triangulated surfaces: WSS trajectories, WSS exposure time (WSSET), fixed
points and stable/unstable manifolds (WSS LCS).

Python / VTK re-implementation of the C++ code used in
A. Arzani, A. M. Gambaruto, G. Chen, S. C. Shadden, "Wall shear stress
exposure time: a Lagrangian measure of near-wall stagnation and concentration
in cardiovascular flows".
"""
from .params import Parameters, FLAG_CODES, write_template
from .mesh import SurfaceMesh
from .field import SurfaceVectorField, FieldSequence
from .fixedpoints import find_fixed_points, FixedPoint, TYPE_NAMES
from .manifolds import compute_manifolds, ManifoldBranch
from .tracer import Particles, advect_step, trace_single

__version__ = "1.0.0"
