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

from trk_to_annotation.datatypes import LABEL_EXTRA_BYTES, LABEL_ALIGN_PAD_FIELD


def _extra_bytes(segments: np.ndarray, scalar_names: list) -> int:
    """Extra per-record bytes beyond the 56 fixed id-record fields: 4 per
    generic float32 scalar, plus LABEL_EXTRA_BYTES if bundle-label fields
    are present."""
    has_label = "label_id" in segments.dtype.names
    return 4 * len(scalar_names) + (LABEL_EXTRA_BYTES if has_label else 0)


# ----------------------------
# Utility Functions
# ----------------------------
def length_of_id_chunk(extra_bytes: int) -> int:
    """
    Compute the byte length of a single ID chunk.

    Parameters
    ----------
    extra_bytes : int
        Extra bytes per segment beyond the fixed record fields (start, end,
        streamline, orientation, orientation_color, padding, number_tracts,
        tract_id = 56 bytes), i.e. 4 bytes per generic float32 scalar plus
        LABEL_EXTRA_BYTES (7) when bundle-label fields are present.

    Returns
    -------
    int
        Byte length of the ID chunk.
    """
    return 56 + extra_bytes


def length_of_id_minishard(id_start: int, id_end: int, extra_bytes: int) -> int:
    """
    Compute the byte length of a minishard containing multiple IDs.

    Parameters
    ----------
    id_start : int
        Starting ID index.
    id_end : int
        Ending ID index (exclusive).
    extra_bytes : int
        Extra bytes per segment beyond the fixed record fields (see
        length_of_id_chunk).

    Returns
    -------
    int
        Byte length of the minishard.
    """
    chunk_indices = 24 * (id_end - id_start)
    chunks = (56 + extra_bytes) * (id_end - id_start)
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
    has_label = "label_id" in segments.dtype.names
    extra_bytes = _extra_bytes(segments, scalar_names)

    dtype = np.dtype(
        [
            ("start", "<f4", 3),
            ("end", "<f4", 3),
            ("streamline", "<u4"),
            ("orientation", "<f4", 3),
            *[(name, "<f4") for name in scalar_names],
            *([("label_id", "<u2"), ("label_name", "<u2"),
               ("label_color", "<u1", 3)] if has_label else []),
            ("orientation_color", "<u1", 3),
            ("padding", "u1"),
            *([LABEL_ALIGN_PAD_FIELD] if has_label else []),
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
    if has_label:
        data["label_id"] = masked_segments["label_id"]
        data["label_name"] = masked_segments["label_name"]
        data["label_color"] = masked_segments["label_color"]
    data["orientation_color"] = np.abs(masked_segments["orientation"] * 255)
    data["padding"] = np.zeros(data.shape[0], dtype="u1")
    data["number_tracts"] = 0
    data["tract_id"] = masked_segments["streamline"]

    data.tofile(f)

    # Write ID metadata
    np.asarray([id_start], dtype="<u8").tofile(f)
    np.asarray(np.ones((id_end - id_start - 1)), dtype="<u8").tofile(f)
    np.asarray(
        [length_of_id_minishard(0, id_start, extra_bytes)], dtype="<u8"
    ).tofile(f)
    np.asarray(np.zeros((id_end - id_start - 1)), dtype="<u8").tofile(f)
    np.asarray(
        [length_of_id_chunk(extra_bytes)] * (id_end - id_start), dtype="<u8"
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
    extra_bytes = _extra_bytes(segments, scalar_names)
    num_ids = len(segments)
    minishard_bits = number_of_minishard_bits_ids(num_ids, preshift_bits)
    per_minishard = 2**preshift_bits

    logging.info("Writing ID shard with %d IDs", num_ids)

    # Write minishard index table
    starts = np.arange(0, num_ids, per_minishard)
    ends = np.minimum(starts + per_minishard, num_ids)

    sizes = length_of_id_minishard(starts, ends, extra_bytes)

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
