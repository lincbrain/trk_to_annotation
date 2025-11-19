from math import ceil, log2
import math
import numpy as np
import tensorstore as ts



def length_of_tract_chunk(tract_num, offsets, scalar_size):
    return (48 + 4*scalar_size)*(offsets[tract_num+1]-offsets[tract_num]) + 8


def length_of_tract_minishard(tract_start, tract_end, offsets, scalar_size):
    chunk_indexs = 24*(tract_end - tract_start)
    chunks = (48 + 4*scalar_size) * \
        (offsets[tract_end] - offsets[tract_start]) + \
        8*(tract_end - tract_start)

    return chunk_indexs + chunks


def number_of_minishard_bits_spatical(num_tracts, preshift_bits):
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
    minishard_bits = number_of_minishard_bits_spatical(num_tracts, preshift_bits)
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

def compressed_morton_code(gridpt, grid_size):
    """Converts a grid point to a compressed morton code.
    from cloud-volume"""
    if hasattr(gridpt, "__len__") and len(gridpt) == 0:  # generators don't have len
        return np.zeros((0,), dtype=np.uint32)

    gridpt = np.asarray(gridpt, dtype=np.uint32)
    single_input = False
    if gridpt.ndim == 1:
        gridpt = np.atleast_2d(gridpt)
        single_input = True

    code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
    num_bits = [math.ceil(math.log2(size)) for size in grid_size]
    j = np.uint64(0)
    one = np.uint64(1)

    if sum(num_bits) > 64:
        raise ValueError(
            f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits."
        )

    max_coords = np.max(gridpt, axis=0)
    if np.any(max_coords >= grid_size):
        raise ValueError(
            f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}"
        )

    for i in range(max(num_bits)):
        for dim in range(3):
            if 2**i < grid_size[dim]:
                bit = ((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j
                code |= bit
                j += one

    if single_input:
        return code[0]
    return code

def write_spacial_shard_2(path, spatial_index, grid_size):
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "murmurhash3_x86_128",
            "preshift_bits": 12,
            "minishard_bits": number_of_minishard_bits_spatical(grid_size**3, 12),
            "shard_bits": 0,
            "minishard_index_encoding": "raw",
            "data_encoding": "raw",
        },
        "base": f"file://{path}",
    }

    dataset = ts.KvStore.open(spec).result()
    txn = ts.Transaction()

    for cell_key, annotations in spatial_index.items():
        value = np.asarray(0, dtype='<u8').tobytes()
        if len(annotations) > 0:
            scalar_names = [name for name in annotations.dtype.names if name.startswith("scalar_")]

            dtype = np.dtype([
                ("start", "<f4", 3),
                ("end", "<f4", 3),
                ("orientation", "<f4", 3),
                *[(name, "<f4", 1) for name in scalar_names],
                ("orientation_color", "<u1", 3),
                ("padding", "<u1", 1),
            ])
            data = np.zeros(len(annotations), dtype=dtype)
            data["start"], data["end"] = annotations["start"], annotations["end"]
            data["orientation"] = annotations["orientation"]
            for name in scalar_names:
                data[name] = annotations[name]
            data["orientation_color"] = np.abs(annotations["orientation"] * 255)
            data["padding"] = 0

            value = (
                np.array(data.shape[0], dtype="<u8").tobytes() +
                data.tobytes() +
                np.array(annotations["id"], dtype="<u8").tobytes()
            )
            print(grid_size)
        index = np.array(cell_key.split("_"), dtype=int)
        mortoncode = compressed_morton_code(index, np.array([grid_size]*3, dtype=np.int32))
        chunk_key = np.ascontiguousarray(mortoncode, dtype=">u8").tobytes()
        dataset.with_transaction(txn)[chunk_key] = value

    txn.commit_async().result()
