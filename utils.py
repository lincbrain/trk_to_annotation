from matplotlib.colors import hsv_to_rgb
import nibabel
import numpy as np
import psutil
import os
import json
import random
WORLD_SPACE_DIMENSION = 1
LIMIT = 4000000


np.random.seed(0)


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
    tracts = nibabel.streamlines.load(trk_file)
    all_streamlines = tracts.tractogram.streamlines
    
    #get the lower bound and uper bound of all the points in the track file
    lb = np.min(all_streamlines._data, axis=0)
    ub = np.max(all_streamlines._data, axis=0)
    print(f"Total number of streamlines: {len(all_streamlines)}")
    log_resource_usage("After Loading Streamlines")

    #get every start point for each line in each tract (remove the last point from each tract)
    streamline_start = np.concatenate([np.array(sl)[:-1] for sl in all_streamlines], axis=0)
    #get every end point for each line in each tract (remove the first point from each tract)
    streamline_end = np.concatenate([np.array(sl)[1:] for sl in all_streamlines], axis=0)
    #index to indicate which tract each line corisponds to
    streamline_tract = np.concatenate([np.full(all_streamlines[i].shape[0]-1, i) for i in range(len(all_streamlines))])

    return streamline_start, streamline_end, streamline_tract, lb, ub

#create new lines everytime a line crosses a grid line of the finest grid 
def split_along_grid(streamline_start, streamline_end, streamline_tract, lb, ub, grid_densities):
    dimensions = ub-lb
    #for each axis (x, y, z)
    for a in range(3):
        #find out where each point is in grid cordinates (both rounded down and not rounded)
        finest_cells_start = np.floor(((streamline_start - lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)
        finest_cells_end = np.floor(((streamline_end-lb)/dimensions)*([grid_densities[-1]]*3)).astype(int)

        #for each line split it into two lines where the point connecting them is a nan point that can be easily identified and removed later
        #these points are a placeholder points incase this line travels across the grid and we need to split it into two lines
        finest_cells_start_not_rounded = np.repeat(np.expand_dims(((streamline_start - lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2,axis=1)
        finest_cells_start_not_rounded[:, 1] = [np.nan, np.nan, np.nan]
        finest_cells_end_not_rounded = np.repeat(np.expand_dims(((streamline_end-lb)/dimensions)*([grid_densities[-1]]*3), axis=1), 2, axis=1)
        finest_cells_end_not_rounded[:, 0] = [np.nan, np.nan, np.nan]

        #create placeholder tract indexes for these placeholder points. make them -1 so they can be easily identified and removed later
        streamline_tract_expand = np.repeat(np.expand_dims(streamline_tract, axis=1), 2, axis=1)
        streamline_tract_expand[:, 1] = -1

        #check to see if the line crosses a grid line of the axis we are currently looking at
        not_same_grid = np.invert(np.equal(finest_cells_start[:, a], finest_cells_end[:, a]))

        #get all the lines that cross a grid line
        cross_grid_end = finest_cells_end_not_rounded[not_same_grid]
        cross_grid_start = finest_cells_start_not_rounded[not_same_grid]

        #replace the placeholder tract index with an actual value for each line that crosses a grid line
        streamline_tract_expand[not_same_grid] = np.repeat(np.expand_dims(streamline_tract_expand[not_same_grid][:, 0], axis=1), 2, axis=1)
        
        #get the value of the grid line being crossed (ie line (1.3, 1.5, 1.5) -> (2.5, 1.5, 1.5) will return 2 for a=0)
        cross_value = np.maximum(finest_cells_end[not_same_grid][:, a], finest_cells_start[not_same_grid][:, a])
        
        #calculate where the line intersects this grid line and assign the placeholder point this value.
        dist_start = np.abs(cross_grid_start[:, 0, a] - cross_value)
        dist_end = np.abs(cross_grid_end[:, 1, a] - cross_value)
        avg_val_1 = (dist_end*cross_grid_start[:, 0, (a + 1)%3] + dist_start*cross_grid_end[:, 1, (a+1)%3])/(dist_start + dist_end)
        avg_val_2 = (dist_end*cross_grid_start[:, 0, (a+2)%3] + dist_start*cross_grid_end[:, 1, (a+2)%3])/(dist_start + dist_end)
        cross_grid_end[:, 0, (a+1)%3] = avg_val_1
        cross_grid_end[:, 0, (a+2)%3] = avg_val_2
        cross_grid_end[:, 0, a] = cross_value
        cross_grid_start[:, 1, (a+1)%3] = avg_val_1
        cross_grid_start[:, 1, (a+2)%3] = avg_val_2
        cross_grid_start[:, 1, a] = cross_value
        finest_cells_end_not_rounded[not_same_grid] = cross_grid_end
        finest_cells_start_not_rounded[not_same_grid] = cross_grid_start


        #remove all placeholder points and then add these points to streamline_tract/start/end
        streamline_tract = np.reshape(streamline_tract_expand, (-1))
        streamline_tract = streamline_tract[streamline_tract != -1]
        streamline_end = np.reshape(finest_cells_end_not_rounded, (-1, 3))
        streamline_end = streamline_end[np.invert(np.isnan(streamline_end[:, 0]))]
        streamline_end = (streamline_end/([grid_densities[-1]]*3))*dimensions + lb
        streamline_start = np.reshape(finest_cells_start_not_rounded, (-1, 3))
        streamline_start = streamline_start[np.invert(np.isnan(streamline_start[:, 0]))]
        streamline_start = (streamline_start/([grid_densities[-1]]*3))*dimensions + lb
    
    #create a list of lines from start points and end points
    lines = np.stack((streamline_start, streamline_end), axis=1)/WORLD_SPACE_DIMENSION

    return lines, streamline_tract

#for each tract write all the lines in that tract to a file in the same format as discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#multiple-annotation-encoding
def write_tract_file(streamline_tract, lines, tract_dir):
    ids = np.arange(0, len(lines))
    tract_id = 0
    while np.any(streamline_tract == tract_id):
        indexes =streamline_tract == tract_id
        tract_file = os.path.join(tract_dir, str(tract_id))
        with open(tract_file, 'wb') as f:
            f.write(np.asarray(len(lines[indexes]), dtype='<u8').tobytes())
            for start, end in lines[indexes]:
                # Write start and end lines as float32
                f.write(np.asarray(start, dtype='<f4').tobytes())
                f.write(np.asarray(end, dtype='<f4').tobytes())
            for annotation_id in ids[indexes]:
                f.write(np.asarray(annotation_id, dtype='<u8').tobytes())  # Write ID as uint64le

        tract_id += 1

#for each line write it to the id file in the format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#single-annotation-encoding
def write_all_lines(lines, id_dir, streamline_tract):
    for i in range(len(lines)):
        id_file = os.path.join(id_dir, str(i))
        with open(id_file, 'wb') as f:
            f.write(np.asarray(lines[i][0], dtype='<f4').tobytes())
            f.write(np.asarray(lines[i][1], dtype='<f4').tobytes())
            f.write(np.asarray([1], dtype='<u4').tobytes())
            f.write(np.asarray(streamline_tract[i], dtype='<u8').tobytes())

#for each spatial level find which lines belong to which sections
#then write them to that section's file
def write_spatial_and_info(lines, lb, ub, grid_densities, streamline_tracts, output_dir):
    grid_shapes = []
    chunk_sizes = []
    ids = np.arange(0, len(lines))
    streamline_start = lines[:, 0]
    dimensions = ub - lb
    #for each spatial level
    for density_index in range(len(grid_densities)):
        spatial_dir = os.path.join(output_dir, str(density_index))
        os.makedirs(spatial_dir, exist_ok=True)
        grid_density = grid_densities[density_index]

        #add new chunk size and grid shape to the list to be used in the info file
        chunk_size = [dim // grid_density for dim in dimensions]
        chunk_sizes.append(chunk_size)
        grid_shape = [grid_density] * 3
        grid_shapes.append(grid_shape)

        #create an empty dictionary that takes a spatial index name and returns three arrays: the lines in the spacial index, the ids of those lines, and the tracts those lines belong to
        spatial_index = {f"{x}_{y}_{z}": [np.array([]),np.array([]),np.array([])] for x in range(grid_shape[0]) for y in range(grid_shape[1]) for z in range(grid_shape[2])}
        cells = np.floor(((streamline_start[:,:3] - lb)/dimensions)*grid_shape).astype(int)
        
        #we need the max amount of lines displayed to see how many tracts we need to filter out to stay within computing limit
        maxAmount = 0

        #fill in that dictonary
        for i in range(grid_shape[0]):
            cells_x = cells[:, 0] == i
            if np.any(cells_x):
                for j in range(grid_shape[1]):
                    cells_xy = cells_x*(cells[:, 1] == j)
                    if np.any(cells_xy):
                        for k in range(grid_shape[2]):
                            cells_xyz = cells_xy*(cells[:, 2] == k)
                            maxAmount = max(maxAmount, len(lines[cells_xyz]))
                            spatial_index[f"{i}_{j}_{k}"] = [np.array(lines[cells_xyz]), np.array(ids[cells_xyz]), np.array(streamline_tracts[cells_xyz])]

        for cell_key, annotations in spatial_index.items():
            #randomly decide what tracts should be kept at this spatial level using a uniform distrabution
            prob = min(LIMIT/maxAmount, 1.0)
            cell_file = os.path.join(spatial_dir, cell_key)
            number_tracts = streamline_tracts[-1] + 1
            rand_selected = np.random.rand(number_tracts) <= prob
            selected_lines = np.isin(annotations[2], np.array(range(number_tracts))[rand_selected])

            #write lines to file using format discribed by https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index
            with open(cell_file, 'wb') as f: 
                if len(annotations[0]) > 0:
                    f.write(np.asarray(len(annotations[0][selected_lines]), dtype='<u8').tobytes())
                    for start, end in annotations[0][selected_lines]:
                        f.write(np.asarray(start, dtype='<f4').tobytes())
                        f.write(np.asarray(end, dtype='<f4').tobytes())
                    for annotation_id in annotations[1][selected_lines]:
                        f.write(np.asarray(annotation_id, dtype='<u8').tobytes())  # Write ID as uint64le
                else:
                    f.write(np.asarray(0, dtype='<u8').tobytes())
            print(f"Saved spatial index for {cell_key} with {len(annotations[0][selected_lines])} annotations on grid density {grid_densities[density_index]}.")

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
        "properties": [],
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