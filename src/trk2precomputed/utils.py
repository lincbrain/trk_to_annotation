import numpy as np

POINT3D_DTYPE = np.dtype([
    ('x', 'f4'),
    ('y', 'f4'),
    ('z', 'f4'),
])

CELL3D_DTYPE = np.dtype([
    ('i', 'i4'),
    ('j', 'i4'),
    ('k', 'i4'),
])

SEGMENT_DTYPE = (
    ('streamline', 'i8'),
    ('start', POINT3D_DTYPE),
    ('end', POINT3D_DTYPE),
    ('orientation', POINT3D_DTYPE),
)


def points2segments(points: np.ndarray) -> np.ndarray:
    """Convert an array of points to an array of segments.

    Parameters
    ----------
    points : np.ndarray
        A vector with structured data type containing
        * streamline : int
          Streamline ID.
        * point : (x: float, y: float, z: float)
          3D coordinates of the points along the streamline.
        * scalar_<name> : float
          Per-point scalars.

    Returns
    -------
    segments : np.ndarray
        A vector with structured data type containing
        * streamline : int
          Streamline ID.
        * start : (x: float, y: float, z: float)
          3D coordinates of the starting point of the segment.
        * end : (x: float, y: float, z: float)
          3D coordinates of the ending point of the segment.
        * scalar_<name> : float
          Per-segment scalar (average of start and end scalars).
        * orient : (dx: float, dy: float, dz: float)
          Orientation vector of the segment (end - start).
    """
    coords = points["points"]
    mask = points["streamline"][:-1] == points["streamline"][1:]

    segment_dtype = list(SEGMENT_DTYPE)
    for name in points.dtype.names:
        if name.startswith("scalar_"):
            segment_dtype.append((name, points.dtype[name]))

    segments = np.zeros([mask.sum()], dtype=segment_dtype)
    segments["streamline"] = points["streamline"][:-1][mask]
    segments["start"] = coords[:-1][mask]
    segments["end"] = coords[1:][mask]

    orient = segments["end"] - segments["start"]
    length = np.linalg.norm(orient, axis=1, keepdims=True)
    orient /= length.clip(min=1e-15)
    segments["orient"] = orient

    for name in points.dtype.names:
        if name.startswith("scalar_"):
            segments[name] = 0.5 * (
                points[name][:-1][mask] + points[name][1:][mask]
            )

    return segments


def insert_boundary_points(
    segments: np.ndarray,
    bbox: np.ndarray,
    grid: list[int],
) -> np.ndarray:
    """
    Insert boundary points into segments that cross grid boundaries.

    Parameters
    ----------
    segments : np.ndarray
        A vector with structured data type containing
        * streamline : int
          Streamline ID.
        * start : (x: float, y: float, z: float)
          3D coordinates of the starting point of the segment.
        * end : (x: float, y: float, z: float)
          3D coordinates of the ending point of the segment.
        * scalar_<name> : float
          Per-segment scalar (average of start and end scalars).
        * orient : (dx: float, dy: float, dz: float)
          Orientation vector of the segment (end - start).
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid : list[int]
        The size of the grid in each dimension (x, y, z).

    Returns
    -------
    np.ndarray
        The segments with boundary points inserted.
        Note that new segments are appended to the end of the array.
        Therefore, segments that were split will not be contiguous anymore.
    """

    for d, size in enumerate(grid):
        boundaries = np.linspace(bbox[0, d], bbox[1, d], size + 1)[1:-1]

        length = np.linalg.norm(segments["start"] - segments["end"], axis=1)
        orient = segments["orient"]

        for boundary in boundaries:
            # Find intersections with the boundary plane
            t = (boundary - segments["start"][d]) / segments["orient"][d]
            mask = (0 < t) & (t < length)
            t = t[mask]

            # create intermediate points
            start = segments["start"][mask]
            end = segments["end"][mask]
            orient = segments["orient"][mask]
            inter = start + t * orient
            segments["end"][mask] = inter

            # create new segments
            new_segments = np.zeros(t.shape, dtype=segments.dtype)
            new_segments["start"] = inter
            new_segments["end"] = end
            new_segments["orient"] = orient

            for name in segments.dtype.names:
                if not name.startswith(("start", "end", "orient")):
                    new_segments[name] = segments[name][mask]

            # Append new segments
            segments = np.concatenate([segments, new_segments])

    return segments


def assign_segments_to_cells(
    segments: np.ndarray,
    bbox: np.ndarray,
    grid: list[int],
) -> np.ndarray:
    """
    Assign segments to grid cells based on their midpoint.

    Parameters
    ----------
    segments : np.ndarray
        A vector with structured data type containing
        * streamline : int
          Streamline ID.
        * start : (x: float, y: float, z: float)
          3D coordinates of the starting point of the segment.
        * end : (x: float, y: float, z: float)
          3D coordinates of the ending point of the segment.
        * scalar_<name> : float
          Per-segment scalar (average of start and end scalars).
        * orient : (dx: float, dy: float, dz: float)
          Orientation vector of the segment (end - start).
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid : list[int]
        The size of the grid in each dimension (x, y, z).

    Returns
    -------
    np.ndarray
        The segments with an additional field 'cell' indicating the grid
        cell index.
    """
    # allocate output with new columns and copy existing data
    cell_dtype = list(segments.dtype) + [("cell", CELL3D_DTYPE)]
    segments_with_cells = np.zeros(segments.shape, dtype=cell_dtype)
    segments_with_cells[segments.dtype.names] = segments

    # assign segments to cells based on midpoints
    # (more robust than endpoints, which may lie exactly on cell boundaries)
    midpoints = 0.5 * (segments["start"] + segments["end"])
    for d, size in enumerate(grid):
        coords = (
            (midpoints[:, d] - bbox[0, d])
            / (bbox[1, d] - bbox[0, d])
            * size
        ).astype(int)
        coords = np.clip(coords, 0, size - 1)
        segments_with_cells["cell"][:, d] = coords

    return segments_with_cells
