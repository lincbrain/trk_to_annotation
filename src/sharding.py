from math import ceil, log2
import numpy as np


def length_of_tract_chunk(tract_num, offsets, scalar_size):
    return (36 + 4*scalar_size)*(offsets[tract_num+1]-offsets[tract_num]) + 8


def length_of_tract_minishard(tract_start, tract_end, offsets, scalar_size):
    chunk_indexs = 24*(tract_end - tract_start)
    chunks = (36 + 4*scalar_size) * \
        (offsets[tract_end] - offsets[tract_start]) + \
        8*(tract_end - tract_start)

    return chunk_indexs + chunks


def number_of_minishard_bits_tracts(num_tracts, preshift_bits):
    return int(ceil(log2(ceil(num_tracts / 2**preshift_bits))))


def write_tract_minishard(tract_start, tract_end, offsets, streamline_scalars, lines, f):
    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("scalars", "<f4", streamline_scalars.shape[1]),
        ("orient_color", "<u1", 3),
        ("padding", "u1", 1),
    ])

    ids = np.arange(0, len(lines))

    # get id of first tract
    np.asarray([tract_start+1], dtype='<u8').tofile(f)
    # add 1 to each subsequent id
    np.asarray(np.ones(((tract_end-tract_start)-1)), dtype='<u8').tofile(f)
    # no start buffer
    np.asarray(np.zeros(((tract_end-tract_start))), dtype='<u8').tofile(f)
    # output size of chunk
    for i in range(tract_start, tract_end):
        np.asarray([length_of_tract_chunk(
            i, offsets, streamline_scalars.shape[1])], dtype='<u8').tofile(f)
    for i in range(tract_start, tract_end):
        index = offsets[i]
        index_end = offsets[i+1]
        data = np.zeros(index_end-index, dtype=dtype)
        tract_line = lines[index:index_end]
        data["start"] = tract_line[:, 0]
        data["end"] = tract_line[:, 1]
        data["scalars"] = streamline_scalars[index:index_end]
        orr = tract_line[:, 1]-tract_line[:, 0]
        tract_line = None
        data["orient_color"] = np.abs(
            orr)/(np.linalg.norm(orr, axis=1).reshape(-1, 1))
        data["padding"] = np.ones(data.shape[0])
        np.asarray(data.shape[0], dtype='<u8').tofile(f)
        data.tofile(f)
        np.asarray(ids[index:index_end], dtype='<u8').tofile(f)


def write_tract_shard(offsets, streamline_scalars, lines, f, preshift_bits=12):
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
        minishard_indexes[index] = last_size
        minishard_indexes[index + 1] = last_size+(tract_end-tract_start)*24
        last_size += length_of_tract_minishard(
            tract_start, tract_end, offsets, streamline_scalars.shape[1])
        tract_start = tract_end
        tract_end += per_minishard
        index = index+2
    minishard_indexes[index:] = last_size
    np.asarray(minishard_indexes, dtype="<u8").tofile(f)

    tract_start = 0
    tract_end = per_minishard
    minishard_indexes = np.zeros(((2**minishard_bits) * 2))
    while tract_start < num_tracts:
        if tract_end > num_tracts:
            tract_end = num_tracts
        write_tract_minishard(tract_start, tract_end,
                              offsets, streamline_scalars, lines, f)
        tract_start = tract_end
        tract_end += per_minishard
