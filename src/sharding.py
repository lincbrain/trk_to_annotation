from math import ceil, log2
import numpy as np
import tensorstore as ts

def number_of_minishard_bits(num_chunks, preshift_bits):
    return int(ceil(log2(ceil(num_chunks / 2**preshift_bits))))

def write_id_shard_2(path, segments):
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
            print(i)
    txn.commit_async().result()

def write_tract_shard_2(path, segments, offsets):
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
