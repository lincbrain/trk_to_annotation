"""
Load .trk file and split line segments where the grid boundaries are

Author: James Scherick
License: Apache-2.0
"""

import logging
from typing import List, Tuple
import nibabel
from nibabel.affines import voxel_sizes as _voxel_sizes
import numpy as np

from trk_to_annotation.datatypes import SEGMENT_DTYPE, LABEL_EXTRA_DTYPE

# ----------------------------
# Configuration
# ----------------------------
BATCH_SIZE = 100_000_000
LABEL_SCALAR_NAME = "label_id"


def label_ids_to_colors(label_ids: np.ndarray) -> np.ndarray:
    """
    Deterministically assign a random RGB color to each unique label id.

    Each id is seeded independently (seed = the id itself), so the color for
    a given label id is stable across batches/runs regardless of which other
    ids are present or the order they're encountered in.

    Parameters
    ----------
    label_ids : np.ndarray
        Integer label ids, one per segment.

    Returns
    -------
    np.ndarray
        (N, 3) uint8 array of RGB colors, one row per input label id.
    """
    unique_ids = np.unique(label_ids)
    lut = np.zeros((int(unique_ids.max()) + 1, 3), dtype=np.uint8)
    for uid in unique_ids:
        rng = np.random.default_rng(int(uid))
        lut[uid] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return lut[label_ids]


def load_from_file(
    trk_file: str, use_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load streamlines from a .trk file.

    Parameters
    ----------
    trk_file : str
        Path to the .trk file.

    Returns
    -------
    segments : np.ndarray
        Structured array containing:
        - streamline : int
          Streamline ID.
        - start : (x, y, z)
          Start point coordinates.
        - end : (x, y, z)
          End point coordinates.
        - scalar_<name> : float
          Per-segment scalar (average of start and end scalars).
        - orientation : (dx, dy, dz)
          Normalized orientation vector.
        - id : int
          id of segment
    bbox : np.ndarray
        Bounding box of the volume as [[x_min, y_min, z_min], [x_max, y_max, z_max]].
    offsets : np.ndarray
        Indices indicating where each streamline starts and ends.
    """
    logging.info("Loading streamlines...")
    tracts = nibabel.streamlines.load(trk_file)
    streamlines = tracts.tractogram.streamlines

    # Transform points to voxel space
    points = np.hstack(
        (streamlines._data[:, :3], np.ones((streamlines._data.shape[0], 1))))
    points = points @ np.linalg.inv(tracts.affine.T)

    # Bounding box
    lb = np.floor(np.min(points, axis=0))[:3]
    ub = np.ceil(np.max(points, axis=0))[:3]
    logging.info(f"Total number of streamlines: {len(streamlines)}")

    # Compute start and end points for segments
    start_idx = np.delete(np.arange(len(points)), np.append(
        streamlines._offsets[1:] - 1, len(points) - 1))
    end_idx = np.delete(np.arange(len(points)), streamlines._offsets)

    line_start = points[start_idx, :3]
    line_end = points[end_idx, :3]

    # Scalar keys (label_id is handled separately below, not as a generic
    # float32 scalar)
    scalar_keys = [
        name for name in tracts.tractogram.data_per_point.keys()
        if name != LABEL_SCALAR_NAME
    ]
    has_label = use_labels and LABEL_SCALAR_NAME in tracts.tractogram.data_per_point.keys()

    segment_dtype = list(SEGMENT_DTYPE)
    for name in scalar_keys:
        segment_dtype.append(("scalar_" + name, "f4"))
    if has_label:
        segment_dtype.extend(LABEL_EXTRA_DTYPE)

    # Streamline IDs
    line_tract = np.concatenate([np.full(length - 1, i + 1)
                                for i, length in enumerate(streamlines._lengths)])

    # Build segments array
    segments = np.zeros(len(line_start), dtype=segment_dtype)
    segments["streamline"] = line_tract
    segments["start"] = line_start
    segments["end"] = line_end
    segments["id"] = np.arange(0, len(line_start))

    # Orientation
    orient = line_end - line_start
    length = np.linalg.norm(orient, axis=1, keepdims=True)
    segments["orientation"] = orient / length.clip(min=1e-15)

    # Scalars
    for i, name in enumerate(scalar_keys):
        segments["scalar_" + name] = np.reshape((tracts.tractogram.data_per_point[name]._data[start_idx] +
                                                 tracts.tractogram.data_per_point[name]._data[end_idx])/2, (-1))

    # Bundle label id / name / color (label_id is constant along a
    # streamline, so start/end average equals the exact value)
    if has_label:
        label_scalar = tracts.tractogram.data_per_point[LABEL_SCALAR_NAME]._data
        label_ids = np.rint(
            (label_scalar[start_idx] + label_scalar[end_idx]) / 2
        ).reshape(-1).astype(np.uint16)
        segments["label_id"] = label_ids
        segments["label_name"] = label_ids
        segments["label_color"] = label_ids_to_colors(label_ids)

    offsets = np.append(streamlines._offsets -
                        np.arange(len(streamlines._offsets)), len(segments))

    logging.info("load_from_file: Done")

    # Points were transformed into voxel-index space above (not world mm),
    # so Neuroglancer needs the per-axis voxel size (mm/index-step) to scale
    # these coordinates back to real physical distances.
    voxel_sizes_mm = _voxel_sizes(tracts.affine)

    return segments, np.array([lb, ub]), offsets, tracts.affine, voxel_sizes_mm


def split_along_grid_batched(
    segments: np.ndarray,
    bbox: np.ndarray,
    grid_densities: List[int],
    offsets: np.ndarray,
    batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split segments into grid-based shards.

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
        * orientation : (dx: float, dy: float, dz: float)
          Orientation vector of the segment (end - start).
        * id : int
          id of segment
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid_densities : list[int]
        The size of the grid in each dimension (x, y, z).
    offsets : np.ndarray
        Array of indices indicating where each streamline starts and ends
    batch_size : int
        size of segment batches to split

    Returns
    -------
    split_segments : np.ndarray
        Segments split along the grid.
    offsets : np.ndarray
        Updated offsets after splitting.
    """

    split_segments = np.zeros(0, dtype=segments.dtype)
    logging.info("Starting grid splitting with %d segments", segments.shape[0])
    split_segments_list = []

    while segments.shape[0] > 0:
        tmp_segments, offsets = split_along_grid(
            segments[:batch_size], bbox, [grid_densities[-1]] * 3, offsets
        )
        split_segments_list.append(tmp_segments)
        segments = segments[batch_size:]
        logging.info("Remaining segments to split: %d", segments.shape[0])
    split_segments = np.concatenate(split_segments_list, axis=0)
    return split_segments, offsets


def split_along_grid(
        segments: np.ndarray,
        bbox: np.ndarray,
        grid: list[int],
        offsets: np.ndarray):
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
        * orientation : (dx: float, dy: float, dz: float)
          Orientation vector of the segment (end - start).
        * id : int
          id of segment
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid : list[int]
        The size of the grid in each dimension (x, y, z).
    offsets : np.ndarray
        Array of indices indicating where each streamline starts and ends

    Returns
    -------
    np.ndarray
        The segments with boundary points inserted.
    np.ndarray
        The new offsets after boundary points are inserted
    """

    offsets_add = np.zeros(offsets.shape)
    # for each axis (x, y, z)
    for d, size in enumerate(grid):
        orient = segments["orientation"]

        repeated_segments = np.repeat(
            np.expand_dims(segments, axis=1), 2, axis=1)
        repeated_segments[:, 1]["start"][:, 0] = np.nan

        cell_start = (segments["start"]-bbox[0])[:, d]//size
        cell_end = (segments["end"]-bbox[0])[:, d]//size

        mask = cell_start != cell_end
        intersection_point = np.maximum.reduce(
            [cell_start[mask], cell_end[mask]])*size + bbox[0][d]

        start = segments["start"][mask]
        orient = segments["orientation"][mask]
        dist = (intersection_point-start[:, d])/orient[:, d]
        inter = start + dist[:, None] * orient
        repeated_segments[:, 0]["end"][mask] = inter
        repeated_segments[:, 1]["start"][mask] = inter
        tracts_added_to = np.bincount(
            repeated_segments[:, 0]["streamline"][mask])
        offsets_add[:tracts_added_to.shape[0]] += tracts_added_to
        segments = repeated_segments.reshape((-1))
        segments = segments[np.invert(np.isnan(segments["start"][:, 0]))]

    offsets = (offsets+np.cumsum(offsets_add)).astype(int)

    return segments, offsets
