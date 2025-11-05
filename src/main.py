import os
import time
from utils import load_from_file, log_resource_usage, split_along_grid, write_all_lines, write_spatial_and_info, write_tract_file


def main(trk_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/sub-I58_sample-hemi_desc-CSD_tractography.smalltest.trk')):
    start_time = time.time()

    output_dir = './precomputed_annotations_new'
    os.makedirs(output_dir, exist_ok=True)
    id_dir = os.path.join(output_dir, 'by_id')
    os.makedirs(id_dir, exist_ok=True)
    tract_dir = os.path.join(output_dir, 'by_tract')
    os.makedirs(tract_dir, exist_ok=True)

    grid_densities = [1, 2, 4, 8, 16]

    streamline_start, streamline_end, streamline_tract, streamline_scalars, scalar_keys, lb, ub = load_from_file(
        trk_file)
    points, streamline_tract, streamline_scalars = split_along_grid(
        streamline_start, streamline_end, streamline_tract, streamline_scalars, lb, ub, grid_densities)
    write_tract_file(streamline_tract, points, streamline_scalars, tract_dir)
    write_all_lines(points, id_dir, streamline_tract, streamline_scalars)
    write_spatial_and_info(points, lb, ub, grid_densities,
                           streamline_tract, streamline_scalars, scalar_keys, output_dir)

    log_resource_usage("After Formatting Annotations")

    # Final metrics
    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")
    log_resource_usage("Final Resource Utilization")


if __name__ == '__main__':
    main()
