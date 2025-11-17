import nibabel
import numpy as np
import psutil
import os
import json

from sharding import number_of_minishard_bits_tracts
WORLD_SPACE_DIMENSION = 1
LIMIT = 50000


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


def load_from_file(trk_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk')):
    """Load streamlines from trk file
    Returns
    ----------
    line_start: shape (n, 3) np array of floats
        stores all the starting points for each annotation
    line_end: shape (n, 3) np array of floats
        stores all the end points for each annotation
    line_tract: shape (n) np array of int
        stores what tract each annotation is in
    line_scalars: shape (n, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    scalar_keys: shape (n) np array of strings
        the keys for each of the scalars that may have been in the tract file
    lb: shape: (3, 3) np array of floats
        stores the lower bound of the annotations
    ub: shape: (3,3) np array of floats
        stores the upper bound of the annotations
    offsets: shape: (number_of_tracts + 1) np array of ints
        the indexs of where every tract starts (and also the value n at the end)
    """
    print("Loading streamlines...")
    tracts = nibabel.streamlines.load(trk_file)
    all_streamlines = tracts.tractogram.streamlines
    points = np.hstack(
        (all_streamlines._data[:, :3], np.ones((all_streamlines._data.shape[0], 1))))
    points = points @ np.linalg.inv(tracts.affine.T)
    # get the lower bound and uper bound of all the points in the track file
    lb = np.array([0, 0, 0])
    ub = np.max(points, axis=0)[:3]
    print(f"Total number of streamlines: {len(all_streamlines)}")
    log_resource_usage("After Loading Streamlines")

    # get every start point for each line in each tract (remove the last point from each tract)
    line_start = np.delete(points, np.append(
        all_streamlines._offsets[1:]-1, len(points)-1), axis=0)

    # get every end point for each line in each tract (remove the first point from each tract)
    line_end = np.delete(
        points, all_streamlines._offsets, axis=0)

    # get every start point for each line in each tract (remove the last point from each tract)
    scalars_start = np.delete(all_streamlines._data[:, 3:], np.append(
        all_streamlines._offsets[1:]-1, len(points)-1), axis=0)

    # get every end point for each line in each tract (remove the first point from each tract)
    scalars_end = np.delete(
        all_streamlines._data[:, 3:], all_streamlines._offsets, axis=0)

    # get additional scalars found in the data
    line_scalars = (scalars_start + scalars_end)/2
    line_start = line_start[:, :3]
    line_end = line_end[:, :3]

    # get scalar keys
    scalar_keys = [key for key in tracts.tractogram.data_per_point.keys()]

    # index to indicate which tract each line corisponds to
    line_tract = np.concatenate([np.full(
        all_streamlines._lengths[i]-1, i) for i in range(len(all_streamlines._lengths))])+1

    return line_start, line_end, line_tract, line_scalars, scalar_keys, lb, ub, np.append(
        all_streamlines._offsets, len(line_start))


def split_along_grid(line_start: np.ndarray, line_end: np.ndarray, line_tract: np.ndarray, line_scalars: np.ndarray, lb: np.ndarray, ub: np.ndarray, offsets, grid_densities: list[int]):
    """Create new lines everytime a line crosses a grid line of the finest grid
    The idea is to first split everywhere it crosses an x grid line, than y, than z
    This is done one at a time because if a line crosses more than one grid line we don't need to figure out where it crosses first. This algorithm will figure that out for us
    The algorithm first creates a bunch of placeholder points working as places to put in memory points if we do find that a line crosses a grid line and needs to be split
    Then find out what lines cross the grid lines and fill in their corisponding placeholder points
    Then remove all placeholder points that were untouched

    Parameters
    ----------
    line_start: shape (n, 3) np array of floats
        stores all the starting points for each annotation
    line_end: shape (n, 3) np array of floats
        stores all the end points for each annotation
    line_tract: shape (n) np array of int
        stores what tract each annotation is in
    line_scalars: shape (n, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    lb: shape: (3, 3) np array of floats
        stores the lower bound of the annotations
    ub: shape: (3,3) np array of floats
        stores the upper bound of the annotations
    offsets: shape: (number_of_tracts + 1) np array of ints
        the indexs of where every tract starts (and also the value n at the end)
    grid_densities: list[int]
        stores how many splits the grids have on each axis. Each number represents a spacial layer, should be increasing, and each should be a power of two

    Returns
    ----------
    lines: shape (m, 2, 3) np array of floats
        stores the starting point and ending point for each annotation
    line_tract: shape (m) np array of ints
        stores what tract each annotation is in
    line_scalars: shape (m, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    offsets: shape: (number_of_tracts + 1) np array of ints
        the indexs of where every tract starts (and also the value m at the end)
    """

    dimensions = ub-lb
    using_scalars = line_scalars.shape[1] > 0
    offsets_add = np.zeros(offsets.shape)
    # for each axis (x, y, z)
    for a in range(3):

        # find out where each point is in grid cordinates (both rounded down and not rounded)
        finest_cells_start = np.floor(
            ((line_start - lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)
        finest_cells_end = np.floor(
            ((line_end-lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)

        # for each line split it into two lines where the point connecting them is a nan point that can be easily identified and removed later
        # these points are placeholder points incase this line travels across the grid and we need to split it into two lines
        finest_cells_start_not_rounded = np.repeat(np.expand_dims(
            ((line_start - lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_start_not_rounded[:, 1] = [np.nan, np.nan, np.nan]
        finest_cells_end_not_rounded = np.repeat(np.expand_dims(
            ((line_end-lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_end_not_rounded[:, 0] = [np.nan, np.nan, np.nan]

        # create placeholder tract indexes for these placeholder points. make them -1 so they can be easily identified and removed later
        line_tract_expand = np.repeat(
            np.expand_dims(line_tract, axis=1), 2, axis=1)
        line_tract_expand[:, 1] = -1

        if using_scalars:
            # create placeholder scalar values for these placeholder points. make them nan so they can be easily identified and removed later
            line_scalars_expand = np.repeat(
                np.expand_dims(line_scalars, axis=1), 2, axis=1)
            line_scalars_expand[:, 1] = np.nan

        # check to see if the line crosses a grid line of the axis we are currently looking at
        not_same_grid = np.invert(
            np.equal(finest_cells_start[:, a], finest_cells_end[:, a]))

        # get all the lines that cross a grid line
        cross_grid_end = finest_cells_end_not_rounded[not_same_grid]
        cross_grid_start = finest_cells_start_not_rounded[not_same_grid]

        tracts_added = np.bincount(
            line_tract_expand[not_same_grid][:, 0]-1)
        offsets_add[:tracts_added.shape[0]] += tracts_added

        # replace the placeholder tract index with an actual value for each line that crosses a grid line
        line_tract_expand[not_same_grid] = np.repeat(np.expand_dims(
            line_tract_expand[not_same_grid][:, 0], axis=1), 2, axis=1)

        if using_scalars:
            # replace the placeholder scalars with an actual value for each line that crosses a grid line
            line_scalars_expand[not_same_grid] = np.repeat(np.expand_dims(
                line_scalars_expand[not_same_grid][:, 0], axis=1), 2, axis=1)

        # get the value of the grid line being crossed (ie line (1.3, 1.5, 1.5) -> (2.5, 1.5, 1.5) will return 2 for a=0)
        cross_value = np.maximum(
            finest_cells_end[not_same_grid][:, a], finest_cells_start[not_same_grid][:, a])

        # calculate where the line intersects this grid line and assign the placeholder point this value.
        dist_start = np.abs(cross_grid_start[:, 0, a] - cross_value)
        dist_end = np.abs(cross_grid_end[:, 1, a] - cross_value)
        avg_val_1 = (dist_end*cross_grid_start[:, 0, (a + 1) % 3] +
                     dist_start*cross_grid_end[:, 1, (a+1) % 3])/(dist_start + dist_end)
        avg_val_2 = (dist_end*cross_grid_start[:, 0, (a+2) % 3] + dist_start *
                     cross_grid_end[:, 1, (a+2) % 3])/(dist_start + dist_end)
        cross_grid_end[:, 0, (a+1) % 3] = avg_val_1
        cross_grid_end[:, 0, (a+2) % 3] = avg_val_2
        cross_grid_end[:, 0, a] = cross_value
        cross_grid_start[:, 1, (a+1) % 3] = avg_val_1
        cross_grid_start[:, 1, (a+2) % 3] = avg_val_2
        cross_grid_start[:, 1, a] = cross_value
        finest_cells_end_not_rounded[not_same_grid] = cross_grid_end
        finest_cells_start_not_rounded[not_same_grid] = cross_grid_start

        # remove all placeholder points and then add these points to line_tract/start/end
        line_tract = np.reshape(line_tract_expand, (-1))
        line_tract = line_tract[line_tract != -1]
        if using_scalars:
            line_scalars = np.reshape(
                line_scalars_expand, (-1, line_scalars.shape[1]))
            line_scalars = line_scalars[np.invert(
                np.isnan(line_scalars[:, 0]))]
        line_end = np.reshape(finest_cells_end_not_rounded, (-1, 3))
        line_end = line_end[np.invert(
            np.isnan(line_end[:, 0]))]
        line_end = (
            line_end/([grid_densities[-1]]*3))*dimensions + lb
        line_start = np.reshape(finest_cells_start_not_rounded, (-1, 3))
        line_start = line_start[np.invert(
            np.isnan(line_start[:, 0]))]
        line_start = (
            line_start/([grid_densities[-1]]*3))*dimensions + lb

    # create a list of lines from start points and end points
    lines = np.stack((line_start, line_end),
                     axis=1)/WORLD_SPACE_DIMENSION

    if not using_scalars:
        line_scalars = np.zeros((line_tract.shape[0], 0))

    return lines, line_tract, line_scalars, (offsets+np.cumsum(offsets_add)).astype(int)


# data type for RGB scalars
rgb_dtype = np.dtype([("r", "<u1"), ("g", "<u1"), ("b", "<u1")])

# data dtype for 3D points
vec3d_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

# data type for the "streamline" or "bundle" relationship
# (each segment is associated to exactly one streamline or exactly one bundle)
relationship_dtype = np.dtype([
    ("nb_objects", "<u4"),      # always 1
    ("object_id", "<u8"),       # streamline id
])


def write_tract_file(lines: np.ndarray, line_scalars: np.ndarray, offsets, tract_dir: str):
    """For each tract write all the lines in that tract to a file in the same format as discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#multiple-annotation-encoding

    Parameters
    ----------
    lines: shape (m, 2, 3) np array of floats
        stores the starting point and ending point for each annotation
    line_scalars: shape (m, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    offsets: shape: (number_of_tracts + 1) np array of ints
        the indexs of where every tract starts (and also the value m at the end)
    tract_dir: string
        the directory where tract files will be saved to
    """

    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("orient", "<f4", 3),
        ("scalars", "<f4", line_scalars.shape[1]),
        ("orient_color", "<u1", 3),
        ("padding", "u1", 1),
    ])

    ids = np.arange(0, len(lines))
    tract_id = 1
    index = 0
    index_end = offsets[1]
    while index < lines.shape[0]:
        data = np.zeros(index_end-index, dtype=dtype)
        tract_line = lines[index:index_end]
        data["start"] = tract_line[:, 0]
        data["end"] = tract_line[:, 1]
        data["scalars"] = line_scalars[index:index_end]
        orr = tract_line[:, 1]-tract_line[:, 0]
        data["orient"] = orr
        tract_line = None
        data["orient_color"] = np.abs(
            orr*255)/(np.linalg.norm(orr, axis=1).reshape(-1, 1))
        data["padding"] = np.ones(data.shape[0])
        # data["streamline"]
        tract_file = os.path.join(tract_dir, str(tract_id))
        with open(tract_file, 'wb') as f:
            np.asarray(data.shape[0], dtype='<u8').tofile(f)
            data.tofile(f)
            np.asarray(ids[index:index_end], dtype='<u8').tofile(f)
        index = index_end
        if index < lines.shape[0]:
            index_end = offsets[tract_id+1]
        tract_id += 1


def write_all_lines(lines: np.ndarray, line_tract: np.ndarray, line_scalars: np.ndarray, id_dir: str):
    """For each line write it to the id file in the format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#single-annotation-encoding

    Parameters
    ----------
    lines: shape (m, 2, 3) np array of floats
        stores the starting point and ending point for each annotation
    line_tract: shape (m) np array of ints
        stores what tract each annotation is in
    line_scalars: shape (m, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    id_dir: string
        the directory where id files will be saved to
    """
    for i in range(len(lines)):
        id_file = os.path.join(id_dir, str(i))
        with open(id_file, 'wb') as f:
            start = lines[i][0]
            end = lines[i][1]
            f.write(np.asarray(start, dtype='<f4').tobytes())
            f.write(np.asarray(end, dtype='<f4').tobytes())
            f.write(np.asarray(line_scalars[i], dtype='<f4').tobytes())
            orr = np.abs(end-start)
            f.write(np.asarray((orr/np.linalg.norm(orr))
                    * 255, dtype='<u1').tobytes())
            f.write(np.asarray([0], dtype='<u1').tobytes())
            f.write(np.asarray([1], dtype='<u4').tobytes())
            f.write(np.asarray(line_tract[i], dtype='<u8').tobytes())


def write_spatial_and_info(lines: np.ndarray, grid_densities: list[int], line_tracts: np.ndarray, line_scalars: np.ndarray, scalar_keys: np.ndarray, lb: np.ndarray, ub: np.ndarray, offsets, output_dir: str):
    """For each spatial level find which lines belong to which sections
    Then write them to that section's file

    Parameters
    ----------
    lines: shape (m, 2, 3) np array of floats
        stores the starting point and ending point for each annotation
    grid_densities: list[int]
        stores how many splits the grids have on each axis. Each number represents a spacial layer, should be increasing, and each should be a power of two
    line_tract: shape (m) np array of ints
        stores what tract each annotation is in
    line_scalars: shape (m, S_n) np array of floats
        stores scalars for each annotation if scalars were in the tract file
    scalar_keys: shape (m) np array of strings
        the keys for each of the scalars that may have been in the tract file
    lb: shape: (3, 3) np array of floats
        stores the lower bound of the annotations
    ub: shape: (3,3) np array of floats
        stores the upper bound of the annotations
    offsets: shape: (number_of_tracts + 1) np array of ints
        the indexs of where every tract starts (and also the value m at the end)
    output_dir: string
        the directory that contains all the files that will be written to including the info file

    """
    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("orient", "<f4", 3),
        ("scalars", "<f4", line_scalars.shape[1]),
        ("orient_color", "<u1", 3),
        # one padding byte because one RGB scalar
        ("padding", "u1", 1),
    ])

    grid_shapes = []
    chunk_sizes = []
    ids = np.arange(0, len(lines))
    dimensions = ub - lb
    tract_level = np.full(line_tracts[-1], -1)
    rand_values = np.random.rand(line_tracts[-1])
    for density_index in range(len(grid_densities)):
        prob = min(
            LIMIT/(lines.shape[0]/(grid_densities[density_index]**3)), 1.0)
        tract_level[rand_values <= prob] = density_index
        rand_values[rand_values <= prob] = 2.0

    # for each spatial level
    for density_index in range(len(grid_densities)):
        spatial_dir = os.path.join(output_dir, str(density_index))
        os.makedirs(spatial_dir, exist_ok=True)
        grid_density = grid_densities[density_index]

        selected_tracts = np.array(
            range(line_tracts[-1]))[tract_level == density_index]

        if len(selected_tracts) > 0:
            lines_list = []
            ids_list = []
            tracts_list = []
            scalars_list = []

            for tract in selected_tracts:
                offset_start = offsets[tract]
                offset_end = offsets[tract + 1]
                lines_list.append(lines[offset_start:offset_end])
                ids_list.append(ids[offset_start:offset_end])
                tracts_list.append(line_tracts[offset_start:offset_end])
                scalars_list.append(
                    line_scalars[offset_start:offset_end])

            # Concatenate once per array
            lines_tmp = np.concatenate(lines_list, axis=0)
            ids_tmp = np.concatenate(ids_list, axis=0)
            tracts_tmp = np.concatenate(tracts_list, axis=0)
            scalars_tmp = np.concatenate(scalars_list, axis=0)

            # add new chunk size and grid shape to the list to be used in the info file
            chunk_size = [dim // grid_density for dim in dimensions]
            chunk_sizes.append(chunk_size)
            grid_shape = [grid_density] * 3
            grid_shapes.append(grid_shape)

            # create an empty dictionary that takes a spatial index name and returns three arrays: the lines in the spacial index, the ids of those lines, the tracts those lines belong to, and any additional scalar values for those lines
            spatial_index = {f"{x}_{y}_{z}": [np.array([]), np.array([]), np.array([]), np.array(
                [])] for x in range(grid_shape[0]) for y in range(grid_shape[1]) for z in range(grid_shape[2])}
            cells = np.floor(
                (((lines_tmp[:, 0] + lines_tmp[:, 1])/2 - lb)/dimensions)*grid_shape).astype(int)

            # fill in that dictonary
            for i in range(grid_shape[0]):
                cells_x = cells[:, 0] == i
                if np.any(cells_x):
                    lines_x = lines_tmp[cells_x]
                    ids_x = ids_tmp[cells_x]
                    tracts_x = tracts_tmp[cells_x]
                    scalars_x = scalars_tmp[cells_x]
                    cells_x_vals = cells[cells_x]
                    for j in range(grid_shape[1]):
                        cells_xy = (cells_x_vals[:, 1] == j)
                        if np.any(cells_xy):
                            lines_xy = lines_x[cells_xy]
                            ids_xy = ids_x[cells_xy]
                            tracts_xy = tracts_x[cells_xy]
                            scalars_xy = scalars_x[cells_xy]
                            cells_xy_vals = cells_x_vals[cells_xy]
                            for k in range(grid_shape[2]):
                                cells_xyz = (cells_xy_vals[:, 2] == k)
                                spatial_index[f"{i}_{j}_{k}"] = [
                                    lines_xy[cells_xyz], ids_xy[cells_xyz], tracts_xy[cells_xyz], scalars_xy[cells_xyz]]

            for cell_key, annotations in spatial_index.items():
                # randomly decide what tracts should be kept at this spatial level using a uniform distrabution
                cell_file = os.path.join(spatial_dir, cell_key)

                # write lines to file using format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index
                with open(cell_file, 'wb') as f:
                    if len(annotations[0]) > 0:
                        data = np.zeros(
                            len(annotations[0]), dtype=dtype)
                        start = annotations[0][:, 0]
                        end = annotations[0][:, 1]
                        data["start"] = start
                        data["end"] = end
                        data["scalars"] = annotations[3]
                        orr = end-start
                        data["orient"] = orr
                        data["orient_color"] = np.abs(
                            orr*255)/(np.linalg.norm(orr, axis=1).reshape(-1, 1))
                        data["padding"] = np.ones(data.shape[0])

                        np.asarray(data.shape[0], dtype='<u8').tofile(f)
                        data.tofile(f)
                        np.asarray(
                            annotations[1], dtype='<u8').tofile(f)
                    else:
                        f.write(np.asarray(0, dtype='<u8').tobytes())
                print(
                    f"Saved spatial index for {cell_key} with {len(annotations[0])} annotations on grid density {grid_densities[density_index]}.")

    # Make an ifo file in the format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#info-json-file-format
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {
            "x": [WORLD_SPACE_DIMENSION, "mm"],
            "y": [WORLD_SPACE_DIMENSION, "mm"],
            "z": [WORLD_SPACE_DIMENSION, "mm"]
        },
        "lower_bound": lb.tolist(),
        "upper_bound": ub.tolist(),
        "annotation_type": "LINE",
        "properties": [
                        {
                "id": "orientation_x",
                "type": "float32",
                "description": "Color-coding of the segment orientation",
            },
            {
                "id": "orientation_y",
                "type": "float32",
                "description": "Color-coding of the segment orientation",
            },
            {
                "id": "orientation_z",
                "type": "float32",
                "description": "Color-coding of the segment orientation",
            },
            *[
                {
                    "id": key,
                    "type": "float32",
                }
                for key in scalar_keys
            ],
            {
                "id": "orientation_color",
                "type": "rgb",
                "description": "Color-coding of the segment orientation",
            }
        ],
        "relationships": [
            {
                "id": "tract",
                "key": "./by_tract",
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "hash": "identity",                # let’s use identity for now
                    "preshift_bits": 12,               # chunks_per_minishard = 2**preshift_bits
                    # single shard = all bits used for minishard
                    "minishard_bits": number_of_minishard_bits_tracts(len(offsets)-1, 12),
                    "shard_bits": 0,                   # single shard = all bits used for minishard
                    "minishard_index_encoding": "raw",  # let’s use raw for now
                    "data_encoding": "raw",            # let’s use raw for now
                }

            }
        ],
        "by_id": {"key": "./by_id"},
        "spatial": [
            {
                "key": f"{i}",
                "grid_shape": grid_shapes[i],
                "chunk_size": chunk_sizes[i],
                "limit": LIMIT
            } for i in range(len(grid_shapes))
        ]
    }
    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)
    print(f"Saved info file at {info_file_path}")


def make_segmenation_layer(lines: np.ndarray, resolution: int, line_tracts: np.ndarray, lb: np.ndarray, ub: np.ndarray, chunk_size: int = 128):
    """Make a segmentation layer to go with the annotation layer (used for selecting tracts)
    TODO: STILL NEED TO OPTOMZIE

    Parameters
    ----------
    lines: shape (m, 2, 3) np array of floats
        stores the starting point and ending point for each annotation
    resolution: int
        width, length, and height of each voxel in mm
    line_tract: shape (m) np array of ints
        stores what tract each annotation is in
    lb: shape: (3, 3) np array of floats
        stores the lower bound of the annotations
    ub: shape: (3,3) np array of floats
        stores the upper bound of the annotations
    output_dir: string
        the directory that contains all the files that will be written to including the info file

    """

    output_dir = "precomputed_segmentation"
    dimensions = ub - lb
    d_x = int(dimensions[0]//(resolution*chunk_size) + 1)
    d_y = int(dimensions[1]//(resolution*chunk_size) + 1)
    d_z = int(dimensions[2]//(resolution*chunk_size) + 1)

    grid = np.zeros((d_z*chunk_size, d_y*chunk_size, d_x*chunk_size, 1))
    for i in range(len(lines)):
        p1 = (lines[i][0] - lb)//resolution
        grid[int(p1[2]), int(p1[1]), int(p1[0]), 0] = line_tracts[i]

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint64",
        "mesh": "mesh",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [[chunk_size, chunk_size, chunk_size]],
                "encoding": "raw",
                "key": f"{resolution}_{resolution}_{resolution}",
                "resolution": [resolution*1000000, resolution*1000000, resolution*1000000],
                "size": [d_x*chunk_size, d_y*chunk_size, d_z*chunk_size],
                "voxel_offset": lb.tolist()
            }
        ],
        "type": "segmentation"
    }
    os.makedirs(output_dir, exist_ok=True)

    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)

    scale_path = os.path.join(
        output_dir, f"{resolution}_{resolution}_{resolution}")
    os.makedirs(scale_path, exist_ok=True)

    for i in range(d_x):
        for j in range(d_y):
            for k in range(d_z):
                chunk_path = os.path.join(
                    scale_path, f"{i*chunk_size}-{(i+1)*chunk_size}_{j*chunk_size}-{(j+1)*chunk_size}_{k*chunk_size}-{(k+1)*chunk_size}")
                with open(chunk_path, 'wb') as f:
                    f.write(np.asarray(
                        grid[k*chunk_size:(k+1)*chunk_size, j*chunk_size:(j+1)*chunk_size, i*chunk_size:(i+1)*chunk_size], dtype='<u8').tobytes())
