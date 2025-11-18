import os
import time
from id_sharding import write_id_shard
from tract_sharding import write_tract_shard, write_tract_shard_2
from utils import make_segmenation_layer, load_from_file, log_resource_usage, split_along_grid, write_spatial_and_info
import numpy as np


def main(trk_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk'), output_dir: str = './precomputed_annotations_new', grid_densities: list[int] = [1, 2, 4, 8, 16]):
    """
    Parameters
    ----------
    trk_file: string
        path to the trk file
    output_dir: string
        path to the output directory
    grid_densities: list[int]
        stores how many splits the grids have on each axis. Each number represents a spacial layer, should be increasing, and each should be a power of two
    """

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    id_dir = os.path.join(output_dir, 'by_id')
    os.makedirs(id_dir, exist_ok=True)
    tract_dir = os.path.join(output_dir, 'by_tract')
    os.makedirs(tract_dir, exist_ok=True)

    batchSize = 100000000

    pre_segments, bbox, offsets = load_from_file(trk_file)
    split_segments = np.zeros(0, dtype=pre_segments.dtype)
    while pre_segments.shape[0] > 0:
        tmp_segments, offsets = split_along_grid(pre_segments[0:batchSize], bbox, [grid_densities[-1]]*3, offsets)
        split_segments = np.concatenate((split_segments, tmp_segments))
        pre_segments = pre_segments[batchSize:]
        print(f"batch size: {batchSize}")
        print(f"remaining: {pre_segments.shape[0]}")
    

    id_file = os.path.join(id_dir, "0.shard")
    with open(id_file, 'wb') as f:
        write_id_shard(split_segments, f)

    tract_file = os.path.join(tract_dir, "0.shard")
    with open(tract_file, 'wb') as f:
        write_tract_shard(offsets, split_segments, f)

    write_spatial_and_info(split_segments, bbox, grid_densities, offsets, output_dir)

    make_segmenation_layer(split_segments, 1, bbox)

    log_resource_usage("After Formatting Annotations")

    # Final metrics
    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")
    log_resource_usage("Final Resource Utilization")


if __name__ == '__main__':
    main()
