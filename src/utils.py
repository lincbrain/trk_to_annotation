import nibabel
import numpy as np
import psutil
import os
import json
WORLD_SPACE_DIMENSION = 1
LIMIT = 40000


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


def load_from_file(trk_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk')):
    """Load streamlines from trk file"""
    print("Loading streamlines...")
    tracts = nibabel.streamlines.load(trk_file)
    all_streamlines = tracts.tractogram.streamlines

    # get the lower bound and uper bound of all the points in the track file
    lb = np.min(all_streamlines._data, axis=0)[:3]
    ub = np.max(all_streamlines._data, axis=0)[:3]
    print(f"Total number of streamlines: {len(all_streamlines)}")
    log_resource_usage("After Loading Streamlines")

    # get every start point for each line in each tract (remove the last point from each tract)
    streamline_start = np.delete(all_streamlines._data, np.append(
        all_streamlines._offsets[1:]-1, len(all_streamlines._data)-1), axis=0)

    # get every end point for each line in each tract (remove the first point from each tract)
    streamline_end = np.delete(
        all_streamlines._data, all_streamlines._offsets, axis=0)

    # get additional scalars found in the data
    streamline_scalars = (streamline_start[:, 3:] + streamline_end[:, 3:])/2
    streamline_start = streamline_start[:, :3]
    streamline_end = streamline_end[:, :3]

    # get scalar keys
    scalar_keys = [key for key in tracts.tractogram.data_per_point.keys()]

    # index to indicate which tract each line corisponds to
    streamline_tract = np.concatenate([np.full(
        all_streamlines._lengths[i]-1, i) for i in range(len(all_streamlines._lengths))])

    return streamline_start, streamline_end, streamline_tract, streamline_scalars, scalar_keys, lb, ub


def split_along_grid(streamline_start, streamline_end, streamline_tract, streamline_scalars, lb, ub, grid_densities):
    """Create new lines everytime a line crosses a grid line of the finest grid
    The idea is to first split everywhere it crosses an x grid line, than y, than z
    This is done one at a time because if a line crosses more than one grid line we don't need to figure out where it crosses first. This algorithm will figure that out for us
    The algorithm first creates a bunch of placeholder points working as places to put in memory points if we do find that a line crosses a grid line and needs to be split
    Then find out what lines cross the grid lines and fill in their corisponding placeholder points
    Then remove all placeholder points that were untouched
    """

    dimensions = ub-lb
    using_scalars = streamline_scalars.shape[1] > 0
    # for each axis (x, y, z)
    for a in range(3):
        # find out where each point is in grid cordinates (both rounded down and not rounded)
        finest_cells_start = np.floor(
            ((streamline_start - lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)
        finest_cells_end = np.floor(
            ((streamline_end-lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)

        # for each line split it into two lines where the point connecting them is a nan point that can be easily identified and removed later
        # these points are placeholder points incase this line travels across the grid and we need to split it into two lines
        finest_cells_start_not_rounded = np.repeat(np.expand_dims(
            ((streamline_start - lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_start_not_rounded[:, 1] = [np.nan, np.nan, np.nan]
        finest_cells_end_not_rounded = np.repeat(np.expand_dims(
            ((streamline_end-lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_end_not_rounded[:, 0] = [np.nan, np.nan, np.nan]

        # create placeholder tract indexes for these placeholder points. make them -1 so they can be easily identified and removed later
        streamline_tract_expand = np.repeat(
            np.expand_dims(streamline_tract, axis=1), 2, axis=1)
        streamline_tract_expand[:, 1] = -1

        if using_scalars:
            # create placeholder scalar values for these placeholder points. make them nan so they can be easily identified and removed later
            streamline_scalars_expand = np.repeat(
                np.expand_dims(streamline_scalars, axis=1), 2, axis=1)
            streamline_scalars_expand[:, 1] = np.nan

        # check to see if the line crosses a grid line of the axis we are currently looking at
        not_same_grid = np.invert(
            np.equal(finest_cells_start[:, a], finest_cells_end[:, a]))

        # get all the lines that cross a grid line
        cross_grid_end = finest_cells_end_not_rounded[not_same_grid]
        cross_grid_start = finest_cells_start_not_rounded[not_same_grid]

        # replace the placeholder tract index with an actual value for each line that crosses a grid line
        streamline_tract_expand[not_same_grid] = np.repeat(np.expand_dims(
            streamline_tract_expand[not_same_grid][:, 0], axis=1), 2, axis=1)

        if using_scalars:
            # replace the placeholder scalars with an actual value for each line that crosses a grid line
            streamline_scalars_expand[not_same_grid] = np.repeat(np.expand_dims(
                streamline_scalars_expand[not_same_grid][:, 0], axis=1), 2, axis=1)

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

        # remove all placeholder points and then add these points to streamline_tract/start/end
        streamline_tract = np.reshape(streamline_tract_expand, (-1))
        streamline_tract = streamline_tract[streamline_tract != -1]
        if using_scalars:
            streamline_scalars = np.reshape(
                streamline_scalars_expand, (-1, streamline_scalars.shape[1]))
            streamline_scalars = streamline_scalars[np.invert(
                np.isnan(streamline_scalars[:, 0]))]
        streamline_end = np.reshape(finest_cells_end_not_rounded, (-1, 3))
        streamline_end = streamline_end[np.invert(
            np.isnan(streamline_end[:, 0]))]
        streamline_end = (
            streamline_end/([grid_densities[-1]]*3))*dimensions + lb
        streamline_start = np.reshape(finest_cells_start_not_rounded, (-1, 3))
        streamline_start = streamline_start[np.invert(
            np.isnan(streamline_start[:, 0]))]
        streamline_start = (
            streamline_start/([grid_densities[-1]]*3))*dimensions + lb

    # create a list of lines from start points and end points
    lines = np.stack((streamline_start, streamline_end),
                     axis=1)/WORLD_SPACE_DIMENSION

    if not using_scalars:
        streamline_scalars = np.zeros((streamline_tract.shape[0], 0))

    return lines, streamline_tract, streamline_scalars


def write_tract_file(streamline_tract, lines, streamline_scalars, tract_dir):
    """For each tract write all the lines in that tract to a file in the same format as discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#multiple-annotation-encoding"""
    ids = np.arange(0, len(lines))
    tract_id = 0
    while np.any(streamline_tract == tract_id):
        indexes = streamline_tract == tract_id
        tract_file = os.path.join(tract_dir, str(tract_id))
        with open(tract_file, 'wb') as f:
            f.write(np.asarray(len(lines[indexes]), dtype='<u8').tobytes())
            for i in range(len(lines[indexes])):
                start = lines[i][0]
                end = lines[i][1]
                # Write start and end lines as float32
                f.write(np.asarray(start, dtype='<f4').tobytes())
                f.write(np.asarray(end, dtype='<f4').tobytes())
                f.write(np.asarray(
                    streamline_scalars[i], dtype='<f4').tobytes())
                orr = np.abs(end-start)
                f.write(np.asarray((orr/np.linalg.norm(orr))
                        * 255, dtype='<u1').tobytes())
                f.write(np.asarray([0], dtype='<u1').tobytes())
            for annotation_id in ids[indexes]:
                # Write ID as uint64le
                f.write(np.asarray(annotation_id, dtype='<u8').tobytes())

        tract_id += 1


def write_all_lines(lines, id_dir, streamline_tract, streamline_scalars):
    """For each line write it to the id file in the format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#single-annotation-encoding"""
    for i in range(len(lines)):
        id_file = os.path.join(id_dir, str(i))
        with open(id_file, 'wb') as f:
            start = lines[i][0]
            end = lines[i][1]
            f.write(np.asarray(start, dtype='<f4').tobytes())
            f.write(np.asarray(end, dtype='<f4').tobytes())
            f.write(np.asarray(streamline_scalars[i], dtype='<f4').tobytes())
            orr = np.abs(end-start)
            f.write(np.asarray((orr/np.linalg.norm(orr))
                    * 255, dtype='<u1').tobytes())
            f.write(np.asarray([0], dtype='<u1').tobytes())
            f.write(np.asarray([1], dtype='<u4').tobytes())
            f.write(np.asarray(streamline_tract[i], dtype='<u8').tobytes())


def write_spatial_and_info(lines, lb, ub, grid_densities, streamline_tracts, streamline_scalars, scalar_keys, output_dir):
    """For each spatial level find which lines belong to which sections
    Then write them to that section's file
    """
    grid_shapes = []
    chunk_sizes = []
    ids = np.arange(0, len(lines))
    streamline_start = lines[:, 0]
    streamline_end = lines[:, 1]
    dimensions = ub - lb
    tract_level = np.full(streamline_tracts[-1]+1, -1)
    rand_values = np.random.rand(streamline_tracts[-1]+1)
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

        # add new chunk size and grid shape to the list to be used in the info file
        chunk_size = [dim // grid_density for dim in dimensions]
        chunk_sizes.append(chunk_size)
        grid_shape = [grid_density] * 3
        grid_shapes.append(grid_shape)

        # create an empty dictionary that takes a spatial index name and returns three arrays: the lines in the spacial index, the ids of those lines, the tracts those lines belong to, and any additional scalar values for those lines
        spatial_index = {f"{x}_{y}_{z}": [np.array([]), np.array([]), np.array([]), np.array(
            [])] for x in range(grid_shape[0]) for y in range(grid_shape[1]) for z in range(grid_shape[2])}
        cells = np.floor(
            (((streamline_start + streamline_end)/2 - lb)/dimensions)*grid_shape).astype(int)

        # fill in that dictonary
        for i in range(grid_shape[0]):
            cells_x = cells[:, 0] == i
            if np.any(cells_x):
                for j in range(grid_shape[1]):
                    cells_xy = cells_x*(cells[:, 1] == j)
                    if np.any(cells_xy):
                        for k in range(grid_shape[2]):
                            cells_xyz = cells_xy*(cells[:, 2] == k)
                            spatial_index[f"{i}_{j}_{k}"] = [
                                lines[cells_xyz], ids[cells_xyz], streamline_tracts[cells_xyz], streamline_scalars[cells_xyz]]

        for cell_key, annotations in spatial_index.items():
            # randomly decide what tracts should be kept at this spatial level using a uniform distrabution
            cell_file = os.path.join(spatial_dir, cell_key)
            number_tracts = streamline_tracts[-1] + 1
            selected_lines = np.isin(annotations[2], np.array(
                range(number_tracts))[tract_level == density_index])

            # write lines to file using format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index
            with open(cell_file, 'wb') as f:
                if len(annotations[0]) > 0:
                    f.write(np.asarray(
                        len(annotations[0][selected_lines]), dtype='<u8').tobytes())
                    starts = annotations[0][selected_lines]
                    ends = annotations[0][selected_lines]
                    scalars = annotations[3][selected_lines]
                    for i in range(len(annotations[0][selected_lines])):
                        start = starts[i][0]
                        end = ends[i][1]
                        f.write(np.asarray(start, dtype='<f4').tobytes())
                        f.write(np.asarray(end, dtype='<f4').tobytes())
                        f.write(np.asarray(scalars[i], dtype='<f4').tobytes())
                        orr = np.abs(end-start)
                        f.write(np.asarray((orr/np.linalg.norm(orr))
                                * 255, dtype='<u1').tobytes())
                        f.write(np.asarray([0], dtype='<u1').tobytes())
                    for annotation_id in annotations[1][selected_lines]:
                        # Write ID as uint64le
                        f.write(np.asarray(annotation_id, dtype='<u8').tobytes())
                else:
                    f.write(np.asarray(0, dtype='<u8').tobytes())
            print(
                f"Saved spatial index for {cell_key} with {len(annotations[0][selected_lines])} annotations on grid density {grid_densities[density_index]}.")

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
                "id": "orientation_color",
                "type": "rgb",
                "description": "Color-coding of the segment orientation",
            },
            *[
                {
                    "id": key,
                    "type": "float32",
                }
                for key in scalar_keys
            ]
        ],
        "relationships": [
            {
                "id": "tract",
                "key": "./by_tract"
            }
        ],
        "by_id": {"key": "./by_id"},
        "spatial": [
            {
                "key": f"{i}",
                "grid_shape": grid_shapes[i],
                "chunk_size": chunk_sizes[i],
                "limit": LIMIT
            } for i in range(len(grid_densities))
        ]
    }
    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)
    print(f"Saved info file at {info_file_path}")
