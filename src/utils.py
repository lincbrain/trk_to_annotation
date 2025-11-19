import nibabel
import numpy as np
import psutil
import os
import json
import tensorstore
from cloudvolume import CloudVolume

from sharding import number_of_minishard_bits

WORLD_SPACE_DIMENSION = 1
LIMIT = 50000

SEGMENT_DTYPE = (
    ('streamline', 'i8'),
    ('start', 'f4', 3),
    ('end', 'f4', 3),
    ('orientation', 'f4', 3),
    ('id', 'i8')
)

np.random.seed(0)


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


def log_resource_usage(stage):
    """Function to log resource utilization"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"[{stage}] CPU Usage: {cpu_percent}%")
    print(f"[{stage}] Memory Usage: {memory.percent}% ({memory.used / (1024**2):.2f} MB used / {memory.total / (1024**2):.2f} MB total)")



def load_from_file(
    trk_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk'
    )
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    bbox : np.ndarray
        Bounding box of the volume as [[x_min, y_min, z_min], [x_max, y_max, z_max]].
    offsets : np.ndarray
        Indices indicating where each streamline starts and ends.
    """
    print("Loading streamlines...")
    tracts = nibabel.streamlines.load(trk_file)
    streamlines = tracts.tractogram.streamlines

    # Transform points to voxel space
    points = np.hstack((streamlines._data[:, :3], np.ones((streamlines._data.shape[0], 1))))
    points = points @ np.linalg.inv(tracts.affine.T)

    # Bounding box
    lb = np.array([0, 0, 0])
    ub = np.max(points, axis=0)[:3]
    print(f"Total number of streamlines: {len(streamlines)}")

    # Compute start and end points for segments
    start_idx = np.delete(np.arange(len(points)), np.append(streamlines._offsets[1:] - 1, len(points) - 1))
    end_idx = np.delete(np.arange(len(points)), streamlines._offsets)

    line_start = points[start_idx, :3]
    line_end = points[end_idx, :3]

    # Scalars
    scalars_start = streamlines._data[start_idx, 3:]
    scalars_end = streamlines._data[end_idx, 3:]
    line_scalars = (scalars_start + scalars_end) / 2

    # Scalar keys
    scalar_keys = list(tracts.tractogram.data_per_point.keys())
    segment_dtype = list(SEGMENT_DTYPE)
    for name in scalar_keys:
        segment_dtype.append(("scalar_" + name, "f4"))

    # Streamline IDs
    line_tract = np.concatenate([np.full(length - 1, i + 1) for i, length in enumerate(streamlines._lengths)])

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
        segments["scalar_" + name] = line_scalars[:, i]
    
    offsets = np.append(streamlines._offsets - np.arange(len(streamlines._offsets)), len(segments))

    print("load_from_file: Done")

    return segments, np.array([lb, ub]), offsets



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
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid : list[int]
        The size of the grid in each dimension (x, y, z).
    offsets : np.ndarray
        Array of indicies indicating where each streamline starts and ends

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
        boundaries = np.linspace(bbox[0, d], bbox[1, d], size + 1)[1:-1]

        orient = segments["orientation"]
        for boundary in boundaries:
            repeated_segments = np.repeat(np.expand_dims(segments, axis=1), 2, axis=1)
            repeated_segments[:, 1]["start"][:, 0] = np.nan

            length = np.linalg.norm(segments["start"] - segments["end"], axis=1)

            t = (boundary - segments["start"][:, d]) / segments["orientation"][:, d]
            mask = (0 < t) & (t < length)
            t = t[mask]

            start = segments["start"][mask]
            orient = segments["orientation"][mask]
            inter = start + t[:, None] * orient
            repeated_segments[:, 0]["end"][mask] = inter
            repeated_segments[:, 1]["start"][mask] = inter
            tracts_added_to = np.bincount(
                repeated_segments[:, 0]["streamline"][mask])
            offsets_add[:tracts_added_to.shape[0]] += tracts_added_to
            segments = repeated_segments.reshape((-1))
            segments = segments[np.invert(np.isnan(segments["start"][:, 0]))]

    offsets = (offsets+np.cumsum(offsets_add)).astype(int)

    return segments, offsets


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
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    grid : list[int]
        The list of densities to split the grid into
    offsets : np.ndarray
        Array of indicies indicating where each streamline starts and ends
    output_dir : string
        where the output files should be written to

    """

   # Collect scalar names
    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]

    grid_shapes, chunk_sizes = [], []
    dimensions = bbox[1] - bbox[0]

    # Assign tract levels based on probability
    tract_count = segments["streamline"][-1]
    tract_level = np.full(tract_count, -1)
    rand_values = np.random.rand(tract_count)

    for density_index, grid_density in enumerate(grid_densities):
        prob = min(LIMIT / (segments.shape[0] / (grid_density ** 3)), 1.0)
        mask = rand_values <= prob
        tract_level[mask] = density_index
        rand_values[mask] = 2.0

    # Process each spatial level
    for density_index, grid_density in enumerate(grid_densities):
        spatial_dir = os.path.join(output_dir, str(density_index))
        os.makedirs(spatial_dir, exist_ok=True)

        selected_tracts = np.where(tract_level == density_index)[0]
        segments_tmp = np.concatenate(
            [segments[offsets[t]:offsets[t + 1]] for t in selected_tracts],
            axis=0
        ) if len(selected_tracts) > 0 else np.zeros(0, dtype=segments.dtype)

        # Grid shape and chunk size
        grid_shape = [grid_density] * 3
        chunk_size = [dim // grid_density for dim in dimensions]
        grid_shapes.append(grid_shape)
        chunk_sizes.append(chunk_size)

        # Spatial index dictionary
        spatial_index = {f"{x}_{y}_{z}": np.array([]) for x in range(grid_shape[0])
                         for y in range(grid_shape[1]) for z in range(grid_shape[2])}

        if len(segments_tmp) > 0:
            cells = np.floor(
                (((segments_tmp["start"] + segments_tmp["end"]) / 2 - bbox[0]) / dimensions) * grid_shape
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
            ("orientation", "<f4", 3),
            *[(name, "<f4", 1) for name in scalar_names],
            ("orient_color", "<u1", 3),
            ("padding", "u1", 1),
        ])

        # Write spatial index files
        #write_spacial_shard_2(os.path.abspath(spatial_dir), spatial_index, grid_density)

        for cell_key, annotations in spatial_index.items():
            cell_file = os.path.join(spatial_dir, cell_key)
            with open(cell_file, 'wb') as f:
                if len(annotations) > 0:
                    data = np.zeros(len(annotations), dtype=file_output_dtype)
                    data["start"], data["end"] = annotations["start"], annotations["end"]
                    data["orientation"] = annotations["orientation"]
                    for name in scalar_names:
                        data[name] = annotations[name]
                    data["orient_color"] = np.abs(annotations["orientation"] * 255)
                    data["padding"] = 1

                    np.asarray(len(data), dtype='<u8').tofile(f)
                    data.tofile(f)
                    np.asarray(annotations["id"], dtype='<u8').tofile(f)
                else:
                    f.write(np.asarray(0, dtype='<u8').tobytes())

        print(f"Saved spatial index at density {grid_density}.")

    # Info file for Neuroglancer
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {axis: [WORLD_SPACE_DIMENSION, "mm"] for axis in ["x", "y", "z"]},
        "lower_bound": bbox[0].tolist(),
        "upper_bound": bbox[1].tolist(),
        "annotation_type": "LINE",
        "properties": [
            {"id": "orientation_x", "type": "float32", "description": "Segment orientation"},
            {"id": "orientation_y", "type": "float32", "description": "Segment orientation"},
            {"id": "orientation_z", "type": "float32", "description": "Segment orientation"},
            *[{"id": key, "type": "float32"} for key in scalar_names],
            {"id": "orientation_color", "type": "rgb", "description": "Orientation color"},
        ],
        "relationships": [{
            "id": "tract",
            "key": "./by_tract",
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "preshift_bits": 12,
                "minishard_bits": number_of_minishard_bits(len(offsets) - 1, 12),
                "shard_bits": 0,
                "minishard_index_encoding": "raw",
                "data_encoding": "raw",
            }
        }],
        "by_id": {"key": "./by_id",             
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "preshift_bits": 12,
                "minishard_bits": number_of_minishard_bits(len(segments), 12),
                "shard_bits": 0,
                "minishard_index_encoding": "raw",
                "data_encoding": "raw",
            }},
        "spatial": [
            {"key": str(i), "grid_shape": grid_shapes[i], "chunk_size": chunk_sizes[i], "limit": LIMIT,
            #"sharding": {
            #    "@type": "neuroglancer_uint64_sharded_v1",
            #    "hash": "identity",
            #    "preshift_bits": 12,
            #    "minishard_bits": number_of_minishard_bits_spatical(chunk_sizes[i][0]**3, 12),
            #    "shard_bits": 0,
            #    "minishard_index_encoding": "raw",
            #    "data_encoding": "raw",
            #}
             }
            for i in range(len(grid_shapes))
        ]
    }

    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)
    print(f"Saved info file at {info_file_path}")



def make_segmenation_layer(segments:np.ndarray, resolution: int, bbox: np.ndarray, chunk_size: int = 128):
    """Make a segmentation layer to go with the annotation layer (used for selecting tracts)

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
    resolution: int
        width, length, and height of each voxel in mm
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    """

    output_dir = "precomputed_segmentation"
    dimensions = bbox[1] - bbox[0]
    d_x = int(dimensions[0]//(resolution*chunk_size) + 1)
    d_y = int(dimensions[1]//(resolution*chunk_size) + 1)
    d_z = int(dimensions[2]//(resolution*chunk_size) + 1)

    grid = np.zeros((d_x*chunk_size, d_y*chunk_size, d_z*chunk_size, 1), dtype="u8")
    for i, segment in enumerate(segments):
        p1 = (segment["start"] - bbox[0])//resolution
        grid[int(p1[0]), int(p1[1]), int(p1[2]), 0] = segment["streamline"]

    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'segmentation',
        data_type       = 'uint64',
        encoding        = 'raw', 
        resolution      = [resolution*1000000, resolution*1000000, resolution*1000000],
        voxel_offset    = bbox[0].tolist(),
        mesh            = 'mesh',
        chunk_size      = [chunk_size, chunk_size, chunk_size],
        volume_size     = [d_x*chunk_size, d_y*chunk_size, d_z*chunk_size]
    )

    os.makedirs(output_dir, exist_ok=True)

    vol = CloudVolume(output_dir, info=info, compress=False)
    vol.commit_info()
    vol[0: d_x*chunk_size, 0:d_y*chunk_size, 0: d_z*chunk_size] = grid[:,:,:]
