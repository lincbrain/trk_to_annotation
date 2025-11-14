import os
import time
from sharding import write_tract_shard
from utils import make_segmenation_layer, load_from_file, log_resource_usage, split_along_grid, write_all_lines, write_spatial_and_info, write_tract_file
import numpy as np


def main(trk_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk'), output_dir: str = './precomputed_annotations_new', grid_densities: list[int] = [1, 2, 4, 8, 16, 32]):
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

    streamline_start, streamline_end, streamline_tracts_pre, streamline_scalars_pre, scalar_keys, lb, ub, offsets = load_from_file(
        trk_file)
    lines, streamline_tracts, streamline_scalars, offsets = split_along_grid(
        streamline_start[0:batchSize], streamline_end[0:batchSize], streamline_tracts_pre[0:batchSize], streamline_scalars_pre[0:batchSize], lb, ub, offsets, grid_densities)
    streamline_start = streamline_start[batchSize:]
    streamline_end = streamline_end[batchSize:]
    streamline_tracts_pre = streamline_tracts_pre[batchSize:]
    streamline_scalars_pre = streamline_scalars_pre[batchSize:]

    while streamline_tracts_pre.shape[0] > 0:
        print(f"batch size: {batchSize}")
        print(f"remaining: {streamline_tracts_pre.shape[0]}")

        lines_batch, streamline_tracts_batch, streamline_scalars_batch, offsets = split_along_grid(
            streamline_start[0:batchSize], streamline_end[0:batchSize], streamline_tracts_pre[0:batchSize], streamline_scalars_pre[0:batchSize], lb, ub, offsets, grid_densities)
        streamline_start = streamline_start[batchSize:]
        streamline_end = streamline_end[batchSize:]
        streamline_tracts_pre = streamline_tracts_pre[batchSize:]
        streamline_scalars_pre = streamline_scalars_pre[batchSize:]
        lines = np.concatenate((lines, lines_batch))
        streamline_tracts = np.concatenate(
            (streamline_tracts, streamline_tracts_batch))
        streamline_scalars = np.concatenate(
            (streamline_scalars, streamline_scalars_batch))

    tract_file = os.path.join(tract_dir, "0.shard")
    with open(tract_file, 'wb') as f:
        write_tract_shard(offsets, streamline_scalars, lines, f)
    write_spatial_and_info(lines, grid_densities,
                           streamline_tracts, streamline_scalars, scalar_keys, lb, ub, offsets, output_dir)

    make_segmenation_layer(lines, 1, streamline_tracts, lb, ub)

    log_resource_usage("After Formatting Annotations")

    # Final metrics
    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")
    log_resource_usage("Final Resource Utilization")


if __name__ == '__main__':
    main()
