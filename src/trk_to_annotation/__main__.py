"""
Main entry point for converting trk files to neuroglancer precomputed annotations

Author: James Scherick
License: Apache-2.0
"""

import argparse
import logging
import os
import time
from trk_to_annotation.preprocessing import load_from_file, split_along_grid_batched
from trk_to_annotation.segmentation import make_segmentation_layer
from trk_to_annotation.tract_sharding import write_tract_shard
from trk_to_annotation.id_sharding import write_id_shard
from trk_to_annotation.utils import save_lta, write_spatial_and_info


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main(trk_file: str, output_dir: str, segmentation_output_dir: str, grid_densities: list[int]):
    """
    Parameters
    ----------
    trk_file: string
        path to the trk file
    output_dir: string
        path to the precomputed annotations directory
    segmentation_output_dir: string
        path to the precomputed segmentations directory
    grid_densities: list[int]
        stores how many splits the grids have on each axis. Each number represents a spatial layer, should be increasing, and each should be a power of two
    """

    os.makedirs(output_dir, exist_ok=True)
    id_dir = os.path.join(output_dir, 'by_id')
    os.makedirs(id_dir, exist_ok=True)
    tract_dir = os.path.join(output_dir, 'by_tract')
    os.makedirs(tract_dir, exist_ok=True)

    start_time = time.time()

    pre_segments, bbox, offsets, affine = load_from_file(trk_file)
    split_segments, offsets = split_along_grid_batched(
        pre_segments, bbox, [grid_densities[-1]]*3, offsets)

    logging.info("Writing ID shards...")
    id_file = os.path.join(id_dir, "0.shard")
    with open(id_file, 'wb') as f:
        write_id_shard(split_segments, f)

    logging.info("Writing tract shards...")
    tract_file = os.path.join(tract_dir, "0.shard")
    with open(tract_file, 'wb') as f:
        write_tract_shard(offsets, split_segments, f)

    logging.info("Writing spatial layers and info file...")
    write_spatial_and_info(split_segments, bbox,
                           grid_densities, offsets, output_dir)

    logging.info("Writing transformation to file...")
    save_lta(affine, os.path.join(output_dir, "transform.lta"))

    logging.info("Creating segmentation layer...")
    make_segmentation_layer(split_segments, 1, bbox, segmentation_output_dir)

    end_time = time.time()
    logging.info("Script completed in %.2f seconds.", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert trk files into neuroglancer precomputed annotations"
    )
    parser.add_argument(
        "trk_file",
        type=str,
        help="Path to the input .trk file",
    )
    parser.add_argument(
        "--annotation_output_dir",
        type=str,
        default="./precomputed_annotations",
        help="Output directory for precomputed annotation",
    )
    parser.add_argument(
        "--segmentation_output_dir",
        type=str,
        default="./precomputed_annotations/precomputed_segmentations",
        help="Output directory for precomputed segmentation",
    )
    parser.add_argument(
        "--grid_densities",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Grid densities (powers of two in ascending order)",
    )

    args = parser.parse_args()
    main(args.trk_file, args.annotation_output_dir,
         args.segmentation_output_dir, args.grid_densities)
