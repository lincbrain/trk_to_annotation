"""
This file provides functions to write tractography tracts into
Neuroglancer precomputed annotation relational shard files.

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
def length_of_id_chunk(scalar_size: int) -> int:
    """
    Compute the byte length of a single ID chunk.

    Parameters
    ----------
    scalar_size : int
        Number of scalar fields per segment.

    Returns
    -------
    int
        Byte length of the ID chunk.
    """
    return 52 + 4 * scalar_size


def length_of_id_minishard(id_start: int, id_end: int, scalar_size: int) -> int:
    """
    Compute the byte length of a minishard containing multiple IDs.

    Parameters
    ----------
    id_start : int
        Starting ID index.
    id_end : int
        Ending ID index (exclusive).
    scalar_size : int
        Number of scalar fields per segment.

    Returns
    -------
    int
        Byte length of the minishard.
    """
    chunk_indices = 24 * (id_end - id_start)
    chunks = (52 + 4 * scalar_size) * (id_end - id_start)
    return chunk_indices + chunks


def number_of_minishard_bits_ids(num_ids: int, preshift_bits: int) -> int:
    """
    Compute the number of minishard bits required for ID shards.

    Parameters
    ----------
    num_ids : int
        Total number of IDs.
    preshift_bits : int
        Number of preshift bits used in sharding.

    Returns
    -------
    int
        Number of minishard bits.
    """
    return int(ceil(log2(ceil(num_ids / 2**preshift_bits))))


# ----------------------------
# Shard Writers
# ----------------------------
def write_id_minishard(
    id_start: int, id_end: int, segments: np.ndarray, f: BinaryIO
) -> None:
    """
    Write a minishard containing multiple IDs to file.

    Parameters
    ----------
    id_start : int
        Starting ID index.
    id_end : int
        Ending ID index (exclusive).
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
            ("number_tracts", "<u4"),
            ("tract_id", "<u8"),
        ]
    )

    data = np.zeros(id_end - id_start, dtype=dtype)
    masked_segments = segments[id_start:id_end]

    data["start"] = masked_segments["start"]
    data["end"] = masked_segments["end"]
    data["orientation"] = masked_segments["orientation"]
    data["streamline"] = masked_segments["streamline"]
    for name in scalar_names:
        data[name] = masked_segments[name]
    data["orientation_color"] = np.abs(masked_segments["orientation"] * 255)
    data["padding"] = np.zeros(data.shape[0], dtype="u1")
    data["number_tracts"] = 0
    data["tract_id"] = masked_segments["streamline"]

    data.tofile(f)

    # Write ID metadata
    np.asarray([id_start], dtype="<u8").tofile(f)
    np.asarray(np.ones((id_end - id_start - 1)), dtype="<u8").tofile(f)
    np.asarray(
        [length_of_id_minishard(0, id_start, len(scalar_names))], dtype="<u8"
    ).tofile(f)
    np.asarray(np.zeros((id_end - id_start - 1)), dtype="<u8").tofile(f)
    np.asarray(
        [length_of_id_chunk(len(scalar_names))] * (id_end - id_start), dtype="<u8"
    ).tofile(f)


def write_id_shard(
    segments: np.ndarray, f: BinaryIO, preshift_bits: int = 12
) -> None:
    """
    Write ID shards to file.

    Parameters
    ----------
    segments : np.ndarray
        Structured array of tractography segments.
    f : BinaryIO
        File handle to write shard data.
    preshift_bits : int, optional
        Number of preshift bits used in sharding (default: 12).
    """
    scalar_names = [
        name for name in segments.dtype.names if name.startswith("scalar_")]
    num_ids = len(segments)
    minishard_bits = number_of_minishard_bits_ids(num_ids, preshift_bits)
    per_minishard = 2**preshift_bits

    logging.info("Writing ID shard with %d IDs", num_ids)

    # Write minishard index table
    starts = np.arange(0, num_ids, per_minishard)
    ends = np.minimum(starts + per_minishard, num_ids)

    sizes = length_of_id_minishard(starts, ends, len(scalar_names))

    last_sizes = np.cumsum(sizes)

    minishard_indices = np.zeros((2**minishard_bits) * 2, dtype=np.int64)

    minishard_indices[0:2*len(starts):2] = last_sizes - (ends - starts) * 24
    minishard_indices[1:2*len(starts):2] = last_sizes

    minishard_indices[2*len(starts):] = last_sizes[-1] + 8

    np.asarray(minishard_indices, dtype="<u8").tofile(f)

    # Write minishards
    id_start, id_end = 0, per_minishard
    while id_start < num_ids:
        id_end = min(id_end, num_ids)
        write_id_minishard(id_start, id_end, segments, f)
        id_start, id_end = id_end, id_end + per_minishard

    logging.info("ID shard writing complete.")
