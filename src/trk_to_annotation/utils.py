"""
Utility functions that do not fit another category 

Author: James Scherick
License: Apache-2.0
"""

import logging
import numpy as np
import os
import json

from trk_to_annotation.id_sharding import number_of_minishard_bits_ids
from trk_to_annotation.tract_sharding import number_of_minishard_bits_tracts


# ----------------------------
# Configuration
# ----------------------------
WORLD_SPACE_DIMENSION = 1
LIMIT = 50000


def convert_to_native(data):
    """Convert data to JSON serializable format"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {k: convert_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(v) for v in data]
    else:
        return data


def lta_data(matrix, src="unknown", dst="unknown"):
    """
    Return the string content of a .lta file for a given 4x4 matrix.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 transformation matrix.
    src : str
        Source volume info.
    dst : str
        Destination volume info.

    Returns
    -------
    str
        The formatted .lta file content.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be 4x4")

    lines = []
    lines.append("type      = 1")
    lines.append("nxforms   = 1")
    lines.append("mean      = 0.0000 0.0000 0.0000")
    lines.append("sigma     = 1.0000")
    lines.append("1 4 4")

    # Matrix rows
    for row in matrix:
        lines.append(" ".join(f"{val:.6f}" for val in row))

    # Volume info
    lines.append("src volume info")
    lines.append(src)
    lines.append("dst volume info")
    lines.append(dst)

    return "\n".join(lines)


def save_lta(matrix, filename, src="unknown", dst="unknown"):
    """
    Save a 4x4 transformation matrix to a .lta file.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 transformation matrix.
    filename : str
        Path to the output .lta file.
    src : str
        Source volume info.
    dst : str
        Destination volume info.
    """
    content = lta_data(matrix, src, dst)
    with open(filename, "w") as f:
        f.write(content)


def generate_info_dict(segments: np.ndarray, bbox: np.ndarray, offsets: np.ndarray, grid_densities: list[int], sharding: bool = True) -> dict:
    """generate the info dictionary for Neuroglancer precomputed annotations

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
    offsets : np.ndarray
        Array of indices indicating where each streamline starts and ends
    grid_densities : list[int]
        The list of densities to split the grid into
    sharding : bool
        Whether to include sharding information in the info dictionary

    Returns
    -------
    dict
        The info dictionary for Neuroglancer precomputed annotations

    """
    dimensions = bbox[1] - bbox[0]
    scalar_names = [
        name for name in segments.dtype.names if name.startswith("scalar_")]

    return {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {axis: [WORLD_SPACE_DIMENSION, "mm"] for axis in ["x", "y", "z"]},
        "lower_bound": bbox[0].tolist(),
        "upper_bound": bbox[1].tolist(),
        "annotation_type": "LINE",
        "properties": [
            {"id": "streamline", "type": "uint32",
                "description": "Segment streamline"},
            {"id": "orientation_x", "type": "float32",
                "description": "Segment orientation"},
            {"id": "orientation_y", "type": "float32",
                "description": "Segment orientation"},
            {"id": "orientation_z", "type": "float32",
                "description": "Segment orientation"},
            *[{"id": key, "type": "float32"} for key in scalar_names],
            {"id": "orientation_color", "type": "rgb",
                "description": "Orientation color"},
        ],
        "relationships": [{
            "id": "tract",
            "key": "./by_tract",
            **(
                {
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "hash": "identity",
                        "preshift_bits": 12,
                        "minishard_bits": number_of_minishard_bits_ids(len(offsets) - 1, 12),
                        "shard_bits": 0,
                        "minishard_index_encoding": "raw",
                        "data_encoding": "raw",
                    }
                } if sharding else {}
            )
        }],
        "by_id": {"key": "./by_id",
            **(
                {
                    "sharding": {
                      "@type": "neuroglancer_uint64_sharded_v1",
                      "hash": "identity",
                      "preshift_bits": 12,
                      "minishard_bits": number_of_minishard_bits_tracts(len(segments), 12),
                      "shard_bits": 0,
                      "minishard_index_encoding": "raw",
                      "data_encoding": "raw",
                  }
                } if sharding else {}
            )
        },
        "spatial": [
            {"key": str(
                i), "grid_shape": [grid_density]*3, "chunk_size": (dimensions/grid_density).tolist(), "limit": LIMIT}
            for i, grid_density in enumerate(grid_densities)
        ]
    }


def get_spatials(segments: np.ndarray, bbox: np.ndarray, offsets: np.ndarray, grid_densities: list[int]) -> list[dict[str, bytes]]:
    """For each spatial level find which lines belong to which sections
    Then return a list of dictionaries where each index corresponds to a density level and
    contains a dictionary mapping cell keys to byte buffers of format "x_y_z"

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
    offsets : np.ndarray
        Array of indices indicating where each streamline starts and ends
    grid_densities : list[int]
        The list of densities to split the grid into

    Returns
    -------
    list[dict[str, bytes]]
        A list of dictionaries where each index corresponds to a density level and
        contains a dictionary mapping cell keys to byte buffers of format "x_y_z"

    """
    np.random.seed(0)
    dimensions = bbox[1] - bbox[0]
    tract_count = segments["streamline"][-1]
    tract_level = np.full(tract_count, -1)
    rand_values = np.random.rand(tract_count)
    for density_index, grid_density in enumerate(grid_densities):
        prob = min(LIMIT / (segments.shape[0] / (grid_density ** 3)), 1.0)
        mask = rand_values <= prob
        tract_level[mask] = density_index
        rand_values[mask] = 2.0

    scalar_names = [
        name for name in segments.dtype.names if name.startswith("scalar_")]
    spatial_levels = []
    for density_index, grid_density in enumerate(grid_densities):

        selected_tracts = np.where(tract_level == density_index)[0]
        segments_tmp = np.concatenate(
            [segments[offsets[t]:offsets[t + 1]] for t in selected_tracts],
            axis=0
        ) if len(selected_tracts) > 0 else np.zeros(0, dtype=segments.dtype)

        # Grid shape and chunk size
        grid_shape = [grid_density] * 3

        # Spatial index dictionary
        spatial_index = {f"{x}_{y}_{z}": np.array([]) for x in range(grid_shape[0])
                         for y in range(grid_shape[1]) for z in range(grid_shape[2])}

        if len(segments_tmp) > 0:
            cells = np.floor(
                (((segments_tmp["start"] + segments_tmp["end"]
                   ) / 2 - bbox[0]) / dimensions) * grid_shape
            ).astype(int)

            # Fill spatial index
            for i in range(grid_shape[0]):
                mask_x = cells[:, 0] == i
                if np.any(mask_x):
                    seg_x, cells_x = segments_tmp[mask_x], cells[mask_x]
                    for j in range(grid_shape[1]):
                        mask_xy = cells_x[:, 1] == j
                        if np.any(mask_xy):
                            seg_xy, cells_xy = seg_x[mask_xy], cells_x[mask_xy]
                            for k in range(grid_shape[2]):
                                mask_xyz = cells_xy[:, 2] == k
                                spatial_index[f"{i}_{j}_{k}"] = seg_xy[mask_xyz]
        # Output dtype for Neuroglancer
        file_output_dtype = np.dtype([
            ("start", "<f4", 3),
            ("end", "<f4", 3),
            ("streamline", "<u4"),
            ("orientation", "<f4", 3),
            *[(name, "<f4") for name in scalar_names],
            ("orientation_color", "<u1", 3),
            ("padding", "u1"),
        ])

        spatial_buffers = {}
        # Write spatial index files
        for cell_key, annotations in spatial_index.items():
            if len(annotations) > 0:
                data = np.zeros(len(annotations), dtype=file_output_dtype)
                data["start"], data["end"] = annotations["start"], annotations["end"]
                data["streamline"] = annotations["streamline"]
                data["orientation"] = annotations["orientation"]
                for name in scalar_names:
                    data[name] = annotations[name]
                data["orientation_color"] = np.abs(
                    annotations["orientation"] * 255)
                data["padding"] = 0

                len_bytes = np.asarray(len(data), dtype='<u8').tobytes()
                data_bytes = data.tobytes()
                id_bytes = np.asarray(annotations["id"], dtype='<u8').tobytes()

                # Concatenate into a single variable
                buffer = len_bytes + data_bytes + id_bytes
            else:
                buffer = np.asarray(0, dtype='<u8').tobytes()
            spatial_buffers[cell_key] = buffer
        spatial_levels.append(spatial_buffers)
    return spatial_levels


def write_spatial_and_info(
    segments: np.ndarray,
    bbox: np.ndarray,
    grid_densities: list[int],
    offsets: np.ndarray,
    output_dir: str
) -> None:
    """For each spatial level find which lines belong to which sections
    Then write them to that section's file

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
        The list of densities to split the grid into
    offsets : np.ndarray
        Array of indices indicating where each streamline starts and ends
    output_dir : str
        where the output files should be written

    """

    # Collect scalar names
    spatials = get_spatials(segments, bbox, offsets, grid_densities)

    # Write spatial index files
    for density_index, grid_density in enumerate(grid_densities):
        spatial_dir = os.path.join(output_dir, str(density_index))
        os.makedirs(spatial_dir, exist_ok=True)
        for cell_key, cell in spatials[density_index].items():
            cell_file = os.path.join(spatial_dir, cell_key)
            with open(cell_file, 'wb') as f:
                f.write(cell)
        logging.info(f"Saved spatial index at density {grid_density}.")

    # Info file for Neuroglancer
    info = generate_info_dict(segments, bbox, offsets, grid_densities)

    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)
    logging.info(f"Saved info file at {info_file_path}")
