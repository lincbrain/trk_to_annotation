import os
import time
from utils import load_from_file, log_resource_usage, split_along_grid, write_all_lines, write_spatial_and_info, write_tract_file


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

    streamline_start, streamline_end, streamline_tract, streamline_scalars, scalar_keys, lb, ub = load_from_file(
        trk_file)
    lines, streamline_tract, streamline_scalars = split_along_grid(
        streamline_start, streamline_end, streamline_tract, streamline_scalars, lb, ub, grid_densities)
    write_tract_file(lines, streamline_tract, streamline_scalars, tract_dir)
    write_all_lines(lines, streamline_tract, streamline_scalars, id_dir)
    write_spatial_and_info(lines, lb, ub, grid_densities,
                           streamline_tract, streamline_scalars, scalar_keys, output_dir)

    log_resource_usage("After Formatting Annotations")

    # Final metrics
    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")
    log_resource_usage("Final Resource Utilization")


if __name__ == '__main__':
    main()
