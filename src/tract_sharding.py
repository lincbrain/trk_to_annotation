"""
Shard writing utilities for neuroglancer annotations.

This file provides the functions to write tractography line segments into
neuroglancer precomputed annotation shard files. 

Author: James Scherick
License: Apache-2.0
"""
from math import ceil, log2
import numpy as np



def length_of_tract_chunk(tract_num, offsets, scalar_size):
    return (48 + 4*scalar_size)*(offsets[tract_num+1]-offsets[tract_num]) + 8


def length_of_tract_minishard(tract_start, tract_end, offsets, scalar_size):
    chunk_indexs = 24*(tract_end - tract_start)
    chunks = (48 + 4*scalar_size) * \
        (offsets[tract_end] - offsets[tract_start]) + \
        8*(tract_end - tract_start)

    return chunk_indexs + chunks


def number_of_minishard_bits_tracts(num_tracts, preshift_bits):
    return int(ceil(log2(ceil(num_tracts / 2**preshift_bits))))


def write_tract_minishard(tract_start, tract_end, offsets, segments, f):
    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]

    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("orientation", "<f4", 3),
        *[(name, "<f4", 1) for name in scalar_names],
        ("orientation_color", "<u1", 3),
        ("padding", "u1", 1),
    ])

    for i in range(tract_start, tract_end):
        index = offsets[i]
        index_end = offsets[i+1]
        data = np.zeros(index_end-index, dtype=dtype)
        masked_segments = segments[index:index_end]
        data["start"] = masked_segments["start"]
        data["end"] = masked_segments["end"]
        data["orientation"] = masked_segments["orientation"]
        for name in scalar_names:
            data[name] = masked_segments[name]
        data["orientation_color"] = np.abs(masked_segments["orientation"] * 255)
        data["padding"] = np.reshape(np.ones(data.shape[0]), (-1, 1))
        np.asarray(data.shape[0], dtype='<u8').tofile(f)
        data.tofile(f)
        np.asarray(masked_segments["id"], dtype='<u8').tofile(f)

    # get id of first tract
    np.asarray([tract_start+1], dtype='<u8').tofile(f)
    # add 1 to each subsequent id
    np.asarray(np.ones(((tract_end-tract_start)-1)), dtype='<u8').tofile(f)
    # no start buffer
    np.asarray(np.zeros(((tract_end-tract_start))), dtype='<u8').tofile(f)
    # output size of chunk
    for i in range(tract_start, tract_end):
        np.asarray([length_of_tract_chunk(
            i, offsets, len(scalar_names))], dtype='<u8').tofile(f)


def write_tract_shard(offsets, segments, f, preshift_bits=12):
    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]
    num_tracts = len(offsets)-1
    minishard_bits = number_of_minishard_bits_tracts(num_tracts, preshift_bits)
    per_minishard = 2**preshift_bits
    tract_start = 0
    tract_end = per_minishard
    last_size = 0
    index = 0
    minishard_indexes = np.zeros(((2**minishard_bits) * 2))
    while tract_start < num_tracts:
        if tract_end > num_tracts:
            tract_end = num_tracts
        last_size += length_of_tract_minishard(
            tract_start, tract_end, offsets, len(scalar_names))
        minishard_indexes[index] = last_size-(tract_end-tract_start)*24
        minishard_indexes[index + 1] = last_size
        tract_start = tract_end
        tract_end += per_minishard
        index = index+2
    minishard_indexes[index:] = last_size+8
    np.asarray(minishard_indexes, dtype="<u8").tofile(f)

    tract_start = 0
    tract_end = per_minishard
    minishard_indexes = np.zeros(((2**minishard_bits) * 2))
    while tract_start < num_tracts:
        if tract_end > num_tracts:
            tract_end = num_tracts
        write_tract_minishard(tract_start, tract_end, offsets, segments, f)
        tract_start = tract_end
        tract_end += per_minishard