"""
VTK input / output helpers (all file I/O of the package goes through VTK).
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import vtk
except ImportError:                                  # ParaView's Python: no top-level 'vtk' module
    import vtkmodules.all as vtk
from vtkmodules.util import numpy_support as vnp

# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------

_READERS = {
    ".vtp": vtk.vtkXMLPolyDataReader,
    ".vtu": vtk.vtkXMLUnstructuredGridReader,
    ".ply": vtk.vtkPLYReader,
    ".stl": vtk.vtkSTLReader,
    ".obj": vtk.vtkOBJReader,
}


def read_dataset(path: str) -> vtk.vtkDataSet:
    """Read any supported file and return the VTK data set."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtk":
        reader = vtk.vtkDataSetReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllNormalsOn()
        reader.ReadAllTensorsOn()
        reader.ReadAllFieldsOn()
        reader.ReadAllColorScalarsOn()
    elif ext in _READERS:
        reader = _READERS[ext]()
    else:
        raise ValueError(f"unsupported file type '{ext}' ({path})")
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"no points could be read from {path}")
    return data


def as_triangle_polydata(data: vtk.vtkDataSet) -> vtk.vtkPolyData:
    """Convert a data set to a triangulated vtkPolyData surface."""
    if not data.IsA("vtkPolyData"):
        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(data)
        geom.Update()
        data = geom.GetOutput()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(data)
    tri.PassVertsOff()
    tri.PassLinesOff()
    tri.Update()
    return tri.GetOutput()


def polydata_triangles(pd: vtk.vtkPolyData) -> np.ndarray:
    """(M,3) integer connectivity of the polygons (must be triangles)."""
    polys = pd.GetPolys()
    n = polys.GetNumberOfCells()
    if n == 0:
        return np.zeros((0, 3), dtype=np.int64)
    conn = vnp.vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    offs = vnp.vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    sizes = np.diff(offs)
    if np.any(sizes != 3):
        raise ValueError("polydata contains non-triangle polygons; triangulate first")
    return conn.reshape(-1, 3)


def polydata_points(pd: vtk.vtkPointSet) -> np.ndarray:
    return vnp.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)


def list_arrays(pd: vtk.vtkDataSet) -> Dict[str, List[Tuple[str, int]]]:
    """Names and number of components of the point and cell arrays."""
    out: Dict[str, List[Tuple[str, int]]] = {"point": [], "cell": []}
    for kind, dat in (("point", pd.GetPointData()), ("cell", pd.GetCellData())):
        for i in range(dat.GetNumberOfArrays()):
            arr = dat.GetArray(i)
            if arr is not None:
                out[kind].append((arr.GetName() or f"array{i}", arr.GetNumberOfComponents()))
    return out


def get_point_vectors(pd: vtk.vtkPolyData, name: Optional[str] = "wss",
                      required: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return the (N,3) point vector array called ``name``.  If it is missing the
    active vectors, then any 3-component point array, then a 3-component cell
    array (averaged to the points) are tried."""
    pdata = pd.GetPointData()
    arr = pdata.GetArray(name) if name else None
    used = name
    if arr is None or arr.GetNumberOfComponents() != 3:
        arr = pdata.GetVectors()
        used = arr.GetName() if arr is not None else None
    if arr is None:
        for i in range(pdata.GetNumberOfArrays()):
            cand = pdata.GetArray(i)
            if cand is not None and cand.GetNumberOfComponents() == 3:
                arr, used = cand, cand.GetName()
                break
    if arr is None:
        cdata = pd.GetCellData()
        carr = cdata.GetArray(name) if name else None
        if carr is None or carr.GetNumberOfComponents() != 3:
            carr = cdata.GetVectors()
        if carr is None:
            for i in range(cdata.GetNumberOfArrays()):
                cand = cdata.GetArray(i)
                if cand is not None and cand.GetNumberOfComponents() == 3:
                    carr = cand
                    break
        if carr is not None:
            warnings.warn(f"vector array '{carr.GetName()}' is cell data; averaging to the points")
            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(pd)
            c2p.PassCellDataOn()
            c2p.Update()
            arr = c2p.GetOutput().GetPointData().GetArray(carr.GetName())
            used = carr.GetName()
    if arr is None:
        if required:
            raise ValueError(f"no 3-component vector array (looked for '{name}') found in the file; "
                             f"available arrays: {list_arrays(pd)}")
        return None, None
    if used != name and name:
        warnings.warn(f"vector array '{name}' not found, using '{used}' instead")
    vec = vnp.vtk_to_numpy(arr).astype(np.float64).reshape(-1, 3)
    return vec, used


def read_surface(path: str, vector_array: Optional[str] = "wss", required: bool = True):
    """Read a surface file.  Returns (points, triangles, vectors, vector_name, polydata)."""
    data = read_dataset(path)
    pd = as_triangle_polydata(data)
    pts = polydata_points(pd)
    tris = polydata_triangles(pd)
    if tris.shape[0] == 0:
        raise ValueError(f"{path} does not contain any triangles")
    vec, used = get_point_vectors(pd, vector_array, required=required)
    return pts, tris, vec, used, pd


# ----------------------------------------------------------------------------
# building polydata
# ----------------------------------------------------------------------------

def _vtk_points(points: np.ndarray) -> vtk.vtkPoints:
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    return pts


def _cell_array(sizes: np.ndarray, connectivity: np.ndarray) -> vtk.vtkCellArray:
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    ca = vtk.vtkCellArray()
    ca.SetData(vnp.numpy_to_vtkIdTypeArray(offsets, deep=True),
               vnp.numpy_to_vtkIdTypeArray(np.ascontiguousarray(connectivity, dtype=np.int64), deep=True))
    return ca


def numpy_to_vtk_array(values: np.ndarray, name: str) -> vtk.vtkDataArray:
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        arr = arr.astype(np.int64 if arr.dtype.itemsize > 4 else np.int32)
    elif arr.dtype.kind == "b":
        arr = arr.astype(np.int32)
    else:
        arr = arr.astype(np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    varr = vnp.numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
    varr.SetName(name)
    return varr


def string_array(values: Sequence[str], name: str) -> vtk.vtkStringArray:
    sarr = vtk.vtkStringArray()
    sarr.SetName(name)
    sarr.SetNumberOfValues(len(values))
    for i, v in enumerate(values):
        sarr.SetValue(i, str(v))
    return sarr


def add_arrays(pd: vtk.vtkPolyData, point_arrays: Optional[Dict[str, np.ndarray]] = None,
               cell_arrays: Optional[Dict[str, np.ndarray]] = None,
               field_arrays: Optional[Dict[str, np.ndarray]] = None) -> vtk.vtkPolyData:
    for name, values in (point_arrays or {}).items():
        if isinstance(values, (list, tuple)) and values and isinstance(values[0], str):
            pd.GetPointData().AddArray(string_array(values, name))
        else:
            pd.GetPointData().AddArray(numpy_to_vtk_array(values, name))
    for name, values in (cell_arrays or {}).items():
        if isinstance(values, (list, tuple)) and values and isinstance(values[0], str):
            pd.GetCellData().AddArray(string_array(values, name))
        else:
            pd.GetCellData().AddArray(numpy_to_vtk_array(values, name))
    for name, values in (field_arrays or {}).items():
        pd.GetFieldData().AddArray(numpy_to_vtk_array(np.atleast_1d(values), name))
    return pd


def surface_polydata(points: np.ndarray, triangles: np.ndarray,
                     point_arrays: Optional[Dict[str, np.ndarray]] = None,
                     cell_arrays: Optional[Dict[str, np.ndarray]] = None,
                     field_arrays: Optional[Dict[str, np.ndarray]] = None) -> vtk.vtkPolyData:
    pd = vtk.vtkPolyData()
    pd.SetPoints(_vtk_points(points))
    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    pd.SetPolys(_cell_array(np.full(len(tris), 3), tris.ravel()))
    return add_arrays(pd, point_arrays, cell_arrays, field_arrays)


def points_polydata(points: np.ndarray, point_arrays: Optional[Dict[str, np.ndarray]] = None,
                    field_arrays: Optional[Dict[str, np.ndarray]] = None) -> vtk.vtkPolyData:
    """Point cloud (one vertex cell per point)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pd = vtk.vtkPolyData()
    pd.SetPoints(_vtk_points(points))
    n = len(points)
    pd.SetVerts(_cell_array(np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)))
    return add_arrays(pd, point_arrays, None, field_arrays)


def polylines_polydata(lines: Iterable[np.ndarray],
                       point_arrays: Optional[Dict[str, Iterable[np.ndarray]]] = None,
                       cell_arrays: Optional[Dict[str, Sequence]] = None,
                       field_arrays: Optional[Dict[str, np.ndarray]] = None) -> vtk.vtkPolyData:
    """Polylines from a list of (n_i,3) arrays.  ``point_arrays`` maps a name to a
    list with one (n_i,...) array per line, ``cell_arrays`` maps a name to one
    value per line."""
    lines = [np.asarray(l, dtype=np.float64).reshape(-1, 3) for l in lines]
    sizes = np.array([len(l) for l in lines], dtype=np.int64)
    pts = np.concatenate(lines, axis=0) if lines else np.zeros((0, 3))
    pd = vtk.vtkPolyData()
    pd.SetPoints(_vtk_points(pts))
    if len(lines):
        pd.SetLines(_cell_array(sizes, np.arange(len(pts), dtype=np.int64)))
    parrays = {}
    for name, per_line in (point_arrays or {}).items():
        per_line = [np.asarray(a) for a in per_line]
        parrays[name] = np.concatenate(per_line, axis=0) if per_line else np.zeros(0)
    carrays = {}
    for name, values in (cell_arrays or {}).items():
        if len(values) and isinstance(values[0], str):
            carrays[name] = list(values)
        else:
            carrays[name] = np.asarray(values)
    return add_arrays(pd, parrays, carrays, field_arrays)


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------

def write_polydata(pd: vtk.vtkPolyData, path: str, binary: bool = True) -> str:
    ext = os.path.splitext(path)[1].lower()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if ext == ".vtp":
        w = vtk.vtkXMLPolyDataWriter()
        if binary:
            w.SetDataModeToAppended()
            w.EncodeAppendedDataOff()
        else:
            w.SetDataModeToAscii()
    elif ext == ".vtk":
        w = vtk.vtkPolyDataWriter()
        w.SetFileTypeToASCII()
        try:
            w.SetFileVersion(42)  # legacy 4.2 format is readable by all tools
        except AttributeError:
            pass
    else:
        raise ValueError("output file must end with .vtp or .vtk")
    w.SetFileName(path)
    w.SetInputData(pd)
    w.Write()
    return path


def write_dataset(data: vtk.vtkDataSet, path: str) -> str:
    """Write a general data set (XML writer chosen by the extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtu":
        w = vtk.vtkXMLUnstructuredGridWriter()
    elif ext == ".vtp":
        w = vtk.vtkXMLPolyDataWriter()
    elif ext == ".vtk":
        w = vtk.vtkDataSetWriter()
        w.SetFileTypeToASCII()
    else:
        raise ValueError("unsupported output extension " + ext)
    w.SetFileName(path)
    w.SetInputData(data)
    w.Write()
    return path


def cell_to_point(pd: vtk.vtkPolyData, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Average cell arrays to the points with vtkCellDataToPointData."""
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(pd)
    c2p.Update()
    out = c2p.GetOutput().GetPointData()
    result = {}
    for i in range(out.GetNumberOfArrays()):
        arr = out.GetArray(i)
        if arr is None:
            continue
        if names is None or arr.GetName() in names:
            result[arr.GetName()] = vnp.vtk_to_numpy(arr)
    return result
