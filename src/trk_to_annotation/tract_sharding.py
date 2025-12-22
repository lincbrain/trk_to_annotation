"""
This file provides functions to write tractography line segments into
Neuroglancer precomputed annotation shard files.

Author: James Scherick
License: Apache-2.0
"""

import logging
from math import ceil, log2
from typing import BinaryIO

import numpy as np


# ----------------------------
# Utility Functions
# ----------------------------
def length_of_tract_chunk(tract_num: int, offsets: np.ndarray, scalar_size: int) -> int:
    """
    Compute the byte length of a single tract chunk.

    Parameters
    ----------
    tract_num : int
        Index of the tract.
    offsets : np.ndarray
        Array of indices indicating tract boundaries.
    scalar_size : int
        Number of scalar fields per segment.

    Returns
    -------
    int
        Byte length of the tract chunk.
    """
    return (48 + 4 * scalar_size) * (offsets[tract_num + 1] - offsets[tract_num]) + 8


def length_of_tract_minishard(
    tract_start: int, tract_end: int, offsets: np.ndarray, scalar_size: int
) -> int:
    """
    Compute the byte length of a minishard containing multiple tracts.

    Parameters
    ----------
    tract_start : int
        Starting tract index.
    tract_end : int
        Ending tract index (exclusive).
    offsets : np.ndarray
        Array of indices indicating tract boundaries.
    scalar_size : int
        Number of scalar fields per segment.

    Returns
    -------
    int
        Byte length of the minishard.
    """
    chunk_indices = 24 * (tract_end - tract_start)
    chunks = (
        (48 + 4 * scalar_size) * (offsets[tract_end] - offsets[tract_start])
        + 8 * (tract_end - tract_start)
    )
    return chunk_indices + chunks


def number_of_minishard_bits_tracts(num_tracts: int, preshift_bits: int) -> int:
    """
    Compute the number of minishard bits required for tract shards.

    Parameters
    ----------
    num_tracts : int
        Total number of tracts.
    preshift_bits : int
        Number of preshift bits used in sharding.

    Returns
    -------
    int
        Number of minishard bits.
    """
    return int(ceil(log2(ceil(num_tracts / 2**preshift_bits))))


def tract_bytes(
        tract_ids: list[int],
        offsets: np.ndarray,
        segments: np.ndarray,
        dtype: np.dtype = None,
        scalar_names: np.ndarray = None
):
    if scalar_names is None:
        scalar_names = [
            name for name in segments.dtype.names if name.startswith("scalar_")]
    if dtype is None:
        dtype = np.dtype(
            [
                ("start", "<f4", 3),
                ("end", "<f4", 3),
                ("streamline", "<u4"),
                ("orientation", "<f4", 3),
                *[(name, "<f4") for name in scalar_names],
                ("orientation_color", "<u1", 3),
                ("padding", "u1"),
            ]
        )
    ids = []
    data_list = []
    for tract_id in tract_ids:
        index, index_end = offsets[tract_id-1], offsets[tract_id]
        data = np.zeros(index_end - index, dtype=dtype)
        masked_segments = segments[index:index_end]

        data["start"] = masked_segments["start"]
        data["end"] = masked_segments["end"]
        data["streamline"] = masked_segments["streamline"]
        data["orientation"] = masked_segments["orientation"]
        for name in scalar_names:
            data[name] = masked_segments[name]
        data["orientation_color"] = np.abs(
            masked_segments["orientation"] * 255)
        data["padding"] = np.zeros(data.shape[0], dtype="u1")

        data_list.append(data)
        ids.append(masked_segments["id"])
    if len(data_list) == 0:
        return np.asarray([0], dtype="<u8").tobytes()
    
    data = np.concatenate(data_list)

    return np.asarray(data.shape[0], dtype="<u8").tobytes() + data.tobytes() + np.asarray(np.concatenate(ids), dtype="<u8").tobytes()
    

# ----------------------------
# Shard Writers
# ----------------------------
def write_tract_minishard(
    tract_start: int,
    tract_end: int,
    offsets: np.ndarray,
    segments: np.ndarray,
    f: BinaryIO,
) -> None:
    """
    Write a minishard containing multiple tracts to file.

    Parameters
    ----------
    tract_start : int
        Starting tract index.
    tract_end : int
        Ending tract index (exclusive).
    offsets : np.ndarray
        Array of indices indicating tract boundaries.
    segments : np.ndarray
        Structured array of tractography segments.
    f : BinaryIO
        File handle to write shard data.
    """
    scalar_names = [
        name for name in segments.dtype.names if name.startswith("scalar_")]

    dtype = np.dtype(
        [
            ("start", "<f4", 3),
            ("end", "<f4", 3),
            ("streamline", "<u4"),
            ("orientation", "<f4", 3),
            *[(name, "<f4") for name in scalar_names],
            ("orientation_color", "<u1", 3),
            ("padding", "u1"),
        ]
    )

    for i in range(tract_start, tract_end):
        index, index_end = offsets[i], offsets[i + 1]
        data = np.zeros(index_end - index, dtype=dtype)
        masked_segments = segments[index:index_end]

        data["start"] = masked_segments["start"]
        data["end"] = masked_segments["end"]
        data["streamline"] = masked_segments["streamline"]
        data["orientation"] = masked_segments["orientation"]
        for name in scalar_names:
            data[name] = masked_segments[name]
        data["orientation_color"] = np.abs(
            masked_segments["orientation"] * 255)
        data["padding"] = np.zeros(data.shape[0], dtype="u1")

        np.asarray(data.shape[0], dtype="<u8").tofile(f)
        data.tofile(f)
        np.asarray(masked_segments["id"], dtype="<u8").tofile(f)

    # Write tract IDs and offsets
    np.asarray([tract_start + 1], dtype="<u8").tofile(f)
    np.asarray(np.ones((tract_end - tract_start - 1)), dtype="<u8").tofile(f)
    np.asarray(
        [length_of_tract_minishard(
            0, tract_start, offsets, len(scalar_names))],
        dtype="<u8",
    ).tofile(f)
    np.asarray(np.zeros((tract_end - tract_start - 1)), dtype="<u8").tofile(f)

    # Write chunk sizes
    for i in range(tract_start, tract_end):
        np.asarray(
            [length_of_tract_chunk(i, offsets, len(scalar_names))], dtype="<u8"
        ).tofile(f)


def write_tract_shard(
    offsets: np.ndarray, segments: np.ndarray, f: BinaryIO, preshift_bits: int = 12
) -> None:
    """
    Write tract shards to file.

    Parameters
    ----------
    offsets : np.ndarray
        Array of indices indicating tract boundaries.
    segments : np.ndarray
        Structured array of tractography segments.
    f : BinaryIO
        File handle to write shard data.
    preshift_bits : int, optional
        Number of preshift bits used in sharding (default: 12).
    """
    scalar_names = [
        name for name in segments.dtype.names if name.startswith("scalar_")]
    num_tracts = len(offsets) - 1
    minishard_bits = number_of_minishard_bits_tracts(num_tracts, preshift_bits)
    per_minishard = 2**preshift_bits

    logging.info("Writing tract shard with %d tracts", num_tracts)

    # Write minishard index table
    starts = np.arange(0, num_tracts, per_minishard)
    ends = np.minimum(starts + per_minishard, num_tracts)

    sizes = length_of_tract_minishard(starts, ends, offsets, len(scalar_names))

    last_sizes = np.cumsum(sizes)

    minishard_indices = np.zeros((2**minishard_bits) * 2, dtype=np.int64)

    minishard_indices[0:2*len(starts):2] = last_sizes - (ends - starts) * 24
    minishard_indices[1:2*len(starts):2] = last_sizes

    minishard_indices[2*len(starts):] = last_sizes[-1] + 8

    np.asarray(minishard_indices, dtype="<u8").tofile(f)

    # Write minishards
    tract_start, tract_end = 0, per_minishard
    while tract_start < num_tracts:
        tract_end = min(tract_end, num_tracts)
        write_tract_minishard(tract_start, tract_end, offsets, segments, f)
        tract_start, tract_end = tract_end, tract_end + per_minishard

    logging.info("Tract shard writing complete.")
