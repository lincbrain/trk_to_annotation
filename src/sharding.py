from math import ceil, log2
import math
import numpy as np
import tensorstore as ts

def number_of_minishard_bits(num_chunks, preshift_bits):
    return int(ceil(log2(ceil(num_chunks / 2**preshift_bits))))

def write_id_shard(path, segments):
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "identity",
            "preshift_bits": 12,
            "minishard_bits": number_of_minishard_bits(len(segments) - 1, 12),
            "shard_bits": 0,
            "minishard_index_encoding": "raw",
            "data_encoding": "raw",
        },
        "base": f"file://{path}",
    }

    dataset = ts.KvStore.open(spec).result()
    txn = ts.Transaction()

    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]

    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("orientation", "<f4", 3),
        *[(name, "<f4", 1) for name in scalar_names],
        ("orientation_color", "<u1", 3),
        ("padding", "u1", 1),
        ("number_tracts", "<u4", 1),
        ("tract_id", "<u8", 1),
    ])
    data = np.zeros(len(segments), dtype=dtype)

    data["start"] = segments["start"]
    data["end"] = segments["end"]
    data["orientation"] = segments["orientation"]
    for name in scalar_names:
        data[name] = segments[name]
    data["orientation_color"] = np.clip(np.abs(segments["orientation"] * 255), 0, 255).astype("u1")
    data["padding"] = 0
    data["number_tracts"] = 1
    data["tract_id"] = np.reshape(segments["streamline"], (-1, 1))

    print("starting writting to shards")
    for i in range(len(segments)):
        key = np.array(i, dtype=">u8").tobytes()
        dataset.with_transaction(txn)[key] = data[i].tobytes()
        if i%10000000 == 0:
            print(f"ids sharded: {i}")
    txn.commit_async().result()

def write_tract_shard(path, segments, offsets):
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "identity",
            "preshift_bits": 12,
            "minishard_bits": number_of_minishard_bits(len(offsets) - 1, 12),
            "shard_bits": 0,
            "minishard_index_encoding": "raw",
            "data_encoding": "raw",
        },
        "base": f"file://{path}",
    }

    dataset = ts.KvStore.open(spec).result()
    txn = ts.Transaction()

    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]

    dtype = np.dtype([
        ("start", "<f4", 3),
        ("end", "<f4", 3),
        ("orientation", "<f4", 3),
        *[(name, "<f4", 1) for name in scalar_names],
        ("orientation_color", "<u1", 3),
        ("padding", "<u1", 1),
    ])

    for i in range(len(offsets) - 1):
        key = np.array(i+1, dtype=">u8").tobytes()
        index, index_end = offsets[i], offsets[i + 1]
        data = np.zeros(index_end - index, dtype=dtype)
        masked_segments = segments[index:index_end]

        data["start"] = masked_segments["start"]
        data["end"] = masked_segments["end"]
        data["orientation"] = masked_segments["orientation"]
        for name in scalar_names:
            data[name] = masked_segments[name]
        data["orientation_color"] = np.clip(np.abs(masked_segments["orientation"] * 255), 0, 255).astype("u1")
        data["padding"] = 1

        value = (
            np.array(data.shape[0], dtype="<u8").tobytes() +
            data.tobytes() +
            np.array(masked_segments["id"], dtype="<u8").tobytes()
        )

        dataset.with_transaction(txn)[key] = value

    txn.commit_async().result()

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

def write_spacial_shard(path, spatial_index, grid_size):
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "murmurhash3_x86_128",
            "preshift_bits": 12,
            "minishard_bits": number_of_minishard_bits(grid_size**3, 12),
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
