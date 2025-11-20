"""
This file provides a function to make a neuroglancer precomputed segmentation layer for precomputed annotations. Separate segmentations by streamline 

Author: James Scherick
License: Apache-2.0
"""

import os
from cloudvolume import CloudVolume
import numpy as np


def make_segmenation_layer(segments: np.ndarray, resolution: int, bbox: np.ndarray, output_dir: str, chunk_size: int = 128):
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
        * id : int
          id of segment
    resolution: int
        width, length, and height of each voxel in mm
    bbox : np.ndarray
        The bounding box of the volume, as a 2x3 array:
        [[x_min, y_min, z_min],
         [x_max, y_max, z_max]]
    output_dir : string
        The folder that will contain the precomputed segmentations
    """

    dimensions = bbox[1] - bbox[0]
    d_xyz = np.asarray(np.ceil(dimensions/(resolution*chunk_size)), dtype="u8")

    grid = np.zeros((d_xyz[0]*chunk_size, d_xyz[1] *
                    chunk_size, d_xyz[2]*chunk_size, 1), dtype="u8")
    for i, segment in enumerate(segments):
        p1 = (segment["start"] - bbox[0])//resolution
        grid[int(p1[0]), int(p1[1]), int(p1[2]), 0] = segment["streamline"]

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint64',
        encoding='raw',
        resolution=[resolution*1000000,
                    resolution*1000000, resolution*1000000],
        voxel_offset=bbox[0].tolist(),
        mesh='mesh',
        chunk_size=[chunk_size, chunk_size, chunk_size],
        volume_size=d_xyz*chunk_size
    )

    os.makedirs(output_dir, exist_ok=True)

    vol = CloudVolume(output_dir, info=info, compress=False)
    vol.commit_info()
    vol[0: d_xyz[0]*chunk_size, 0:d_xyz[1]*chunk_size,
        0: d_xyz[2]*chunk_size] = grid[:, :, :]
