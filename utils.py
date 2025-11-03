from matplotlib.colors import hsv_to_rgb
import numpy as np
import psutil
import os
import json
from dipy.io.streamline import load_trk
WORLD_SPACE_DIMENSION = 1

def generate_colors(num_colors):
    hues = np.linspace(0, 1, num_colors, endpoint=False)  # Evenly spaced hues
    saturation = 0.9  # High saturation
    brightness = 0.9  # High brightness
    colors = [hsv_to_rgb([hue, saturation, brightness]) for hue in hues]
    return (np.array(colors) * 255).astype(np.uint8)  # Convert to 8-bit RGB


# Convert data to JSON serializable format
def convert_to_native(data):
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

# Function to log resource utilization
def log_resource_usage(stage):
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"[{stage}] CPU Usage: {cpu_percent}%")
    print(f"[{stage}] Memory Usage: {memory.percent}% ({memory.used / (1024**2):.2f} MB used / {memory.total / (1024**2):.2f} MB total)")

#load streamlines from trk file
def load_from_file(trk_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk')):
    print("Loading streamlines...")
    sft = load_trk(trk_file, reference='same')
    all_streamlines = sft.streamlines
    print(f"Total number of streamlines: {len(all_streamlines)}")
    log_resource_usage("After Loading Streamlines")

    streamline_start = np.concatenate([np.array(sl)[:-1] for sl in all_streamlines], axis=0)
    streamline_end = np.concatenate([np.array(sl)[1:] for sl in all_streamlines], axis=0)
    streamline_tract = np.concatenate([np.full(all_streamlines[i].shape[0]-1, i) for i in range(len(all_streamlines))])

    return streamline_start, streamline_end, streamline_tract, sft

#create new points everytime a line crosses a grid line of the finest grid 
def split_along_grid(streamline_start, streamline_end, streamline_tract, dimensions, affine, grid_densities):
    homogeneous_streamline = np.hstack((streamline_start, np.ones((streamline_start.shape[0], 1))))
    absoluteCords = (homogeneous_streamline @ np.linalg.inv(affine))[:, :3]
    homogeneous_streamline_end = np.hstack((streamline_end, np.ones((streamline_end.shape[0], 1))))
    absoluteCords_end = (homogeneous_streamline_end @ np.linalg.inv(affine))[:, :3]

    for a in range(3):
        finest_cells_start = np.floor((absoluteCords/dimensions)*([grid_densities[-1]]*3)).astype(int)
        finest_cells_start_notRounded = np.repeat(np.expand_dims((absoluteCords/dimensions)*([grid_densities[-1]]*3), axis=1), 2,axis=1)
        finest_cells_start_notRounded[:, 1] = [-1, -1, -1]
        finest_cells_end = np.floor((absoluteCords_end/dimensions)*([grid_densities[-1]]*3)).astype(int)
        finest_cells_end_notRounded = np.repeat(np.expand_dims((absoluteCords_end/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_end_notRounded[:, 0] = [-1, -1, -1]
        streamline_tract_expand = np.repeat(np.expand_dims(streamline_tract, axis=1), 2, axis=1)
        streamline_tract_expand[:, 1] = -1
        sameX = np.invert(np.equal(finest_cells_start[:, a], finest_cells_end[:, a]))

        acrossX_end = finest_cells_end_notRounded[sameX]
        acrossX_start = finest_cells_start_notRounded[sameX]
        streamline_tract_expand[sameX] = np.repeat(np.expand_dims(streamline_tract_expand[sameX][:, 0], axis=1), 2, axis=1)
        xCross = np.maximum(finest_cells_end[sameX][:, a], finest_cells_start[sameX][:, a])
        dist_start = np.abs(acrossX_start[:, 0, a] - xCross)
        dist_end = np.abs(acrossX_end[:, 1, a] - xCross)
        avgY = (dist_end*acrossX_start[:, 0, (a + 1)%3] + dist_start*acrossX_end[:, 1, (a+1)%3])/(dist_start + dist_end)
        avgZ = (dist_end*acrossX_start[:, 0, (a+2)%3] + dist_start*acrossX_end[:, 1, (a+2)%3])/(dist_start + dist_end)
        acrossX_end[:, 0, (a+1)%3] = avgY
        acrossX_end[:, 0, (a+2)%3] = avgZ
        acrossX_end[:, 0, a] = xCross
        acrossX_start[:, 1, (a+1)%3] = avgY
        acrossX_start[:, 1, (a+2)%3] = avgZ
        acrossX_start[:, 1, a] = xCross
        finest_cells_end_notRounded[sameX] = acrossX_end
        finest_cells_start_notRounded[sameX] = acrossX_start

        streamline_tract = np.reshape(streamline_tract_expand, (-1))
        streamline_tract = streamline_tract[streamline_tract != -1]
        absoluteCords_end = np.reshape(finest_cells_end_notRounded, (-1, 3))
        absoluteCords_end = absoluteCords_end[absoluteCords_end[:, 0] != -1]
        absoluteCords_end = (absoluteCords_end/([grid_densities[-1]]*3))*dimensions
        absoluteCords = np.reshape(finest_cells_start_notRounded, (-1, 3))
        absoluteCords = absoluteCords[absoluteCords[:, 0] != -1]
        absoluteCords = (absoluteCords/([grid_densities[-1]]*3))*dimensions
        

    absoluteCords = np.hstack((absoluteCords, np.ones((absoluteCords.shape[0], 1))))
    absoluteCords_end = np.hstack((absoluteCords_end, np.ones((absoluteCords.shape[0], 1))))

    streamline_start = (absoluteCords @ affine)[:,:3]
    streamline_end = (absoluteCords_end @ affine)[:, :3]
    points = np.stack((streamline_start, streamline_end), axis=1)/WORLD_SPACE_DIMENSION

    return points, streamline_tract

def write_tract_file(streamline_tract, points, tract_dir):
    ids = np.arange(0, len(points))
    tract_id = 0
    while np.any(streamline_tract == tract_id):
        indexes =streamline_tract == tract_id
        tract_file = os.path.join(tract_dir, str(tract_id))
        with open(tract_file, 'wb') as f:
            f.write(np.asarray(len(points[indexes]), dtype='<u8').tobytes())
            for start, end in points[indexes]:
                # Write start and end points as float32
                f.write(np.asarray(start, dtype='<f4').tobytes())
                f.write(np.asarray(end, dtype='<f4').tobytes())
            for annotation_id in ids[indexes]:
                f.write(np.asarray(annotation_id, dtype='<u8').tobytes())  # Write ID as uint64le

        tract_id += 1
def write_all_points(points, id_dir):
    for i in range(len(points)):
        id_file = os.path.join(id_dir, str(i))
        with open(id_file, 'wb') as f:
            f.write(np.asarray(points[i][0], dtype='<f4').tobytes())
            f.write(np.asarray(points[i][1], dtype='<f4').tobytes())

def write_spatial_and_info(points, dimensions, affine, grid_densities, output_dir):
    grid_shapes = []
    chunk_sizes = []
    ids = np.arange(0, len(points))
    homogeneous_streamline = np.hstack((points[:, 0], np.ones((points[:, 0].shape[0], 1))))
    absoluteCords = (homogeneous_streamline @ np.linalg.inv(affine))[:, :3]
    for density_index in range(len(grid_densities)):
        spatial_dir = os.path.join(output_dir, str(density_index))
        os.makedirs(spatial_dir, exist_ok=True)
        grid_density = grid_densities[density_index]
        chunk_size = [dim // grid_density for dim in dimensions]
        chunk_sizes.append(chunk_size)
        grid_shape = [grid_density] * 3
        grid_shapes.append(grid_shape)
        spatial_index = {f"{x}_{y}_{z}": [[],[]] for x in range(grid_shape[0]) for y in range(grid_shape[1]) for z in range(grid_shape[2])}
        cells = np.floor((absoluteCords[:,:3]/dimensions)*grid_shape).astype(int)
        for i in range(grid_shape[0]):
            cells_x = cells[:, 0] == i
            if np.any(cells_x):
                for j in range(grid_shape[1]):
                    cells_xy = cells_x*(cells[:, 1] == j)
                    if np.any(cells_xy):
                        for k in range(grid_shape[2]):
                            cells_xyz = cells_xy*(cells[:, 2] == k)
                            spatial_index[f"{i}_{j}_{k}"] = [points[cells_xyz], ids[cells_xyz]]

        for cell_key, annotations in spatial_index.items():
            if len(annotations[0]) > 0:
                cell_file = os.path.join(spatial_dir, cell_key)
                with open(cell_file, 'wb') as f:
                    f.write(np.asarray(len(annotations[0]), dtype='<u8').tobytes())
                    for start, end in annotations[0]:
                        f.write(np.asarray(start, dtype='<f4').tobytes())
                        f.write(np.asarray(end, dtype='<f4').tobytes())
                    for annotation_id in annotations[1]:
                        f.write(np.asarray(annotation_id, dtype='<u8').tobytes())  # Write ID as uint64le
                print(f"Saved spatial index for {cell_key} with {len(annotations[0])} annotations on grid density {grid_densities[density_index]}.")

    lb = np.min(np.min(np.reshape(points, (-1, 3)), axis=0))
    ub = np.max(np.min(np.reshape(points, (-1, 3)), axis=0))
    # Save info file
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
        "properties": [],
        "relationships": [
            {
                "id": "tract",
                "key": "by_tract" 
            }
        ],
        "by_id": {"key": "annotations_by_id"},
        "spatial": [
            {
                "key": f"{i}",
                "grid_shape": grid_shapes[i],
                "chunk_size": chunk_sizes[i],
                "limit": 5000000000
            } for i in range(len(grid_densities))
        ]
    }
    info_file_path = os.path.join(output_dir, 'info')
    with open(info_file_path, 'w') as f:
        json.dump(convert_to_native(info), f)
    print(f"Saved info file at {info_file_path}")