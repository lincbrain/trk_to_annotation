from math import ceil, log2
import numpy as np
import tensorstore as ts



def length_of_id_chunk(scalar_size):
    return (52 + 4*scalar_size)


def length_of_id_minishard(id_start, id_end, scalar_size):
    chunk_indexs = 24*(id_end - id_start)
    chunks = (52 + 4*scalar_size)*(id_end - id_start)

    return chunk_indexs + chunks


def number_of_minishard_bits_ids(num_ids, preshift_bits):
    return int(ceil(log2(ceil(num_ids / 2**preshift_bits))))


def write_id_minishard(id_start, id_end, segments, f):
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

    data = np.zeros(id_end-id_start, dtype=dtype)
    masked_segments = segments[id_start:id_end]
    data["start"] = masked_segments["start"]
    data["end"] = masked_segments["end"]
    data["orientation"] = masked_segments["orientation"]
    for name in scalar_names:
        data[name] = masked_segments[name]
    data["orientation_color"] = np.abs(masked_segments["orientation"] * 255)
    data["padding"] = np.reshape(np.ones(data.shape[0]), (-1, 1))
    data["number_tracts"] = 1
    data["tract_id"] = np.reshape(masked_segments["streamline"], (-1, 1))
    data.tofile(f)
    
    # get id of first id
    np.asarray([id_start+1], dtype='<u8').tofile(f)
    # add 1 to each subsequent id
    np.asarray(np.ones(((id_end-id_start)-1)), dtype='<u8').tofile(f)
    # no start buffer
    np.asarray(np.zeros(((id_end-id_start))), dtype='<u8').tofile(f)
    # output size of chunk
    np.asarray([length_of_id_chunk(len(scalar_names))]*(id_end - id_start), dtype='<u8').tofile(f)


def write_id_shard(segments, f, preshift_bits=12):
    scalar_names = [name for name in segments.dtype.names if name.startswith("scalar_")]
    num_ids = len(segments)
    minishard_bits = number_of_minishard_bits_ids(num_ids, preshift_bits)
    per_minishard = 2**preshift_bits
    id_start = 0
    id_end = per_minishard
    last_size = 0
    index = 0
    minishard_indexes = np.zeros(((2**minishard_bits) * 2))
    while id_start < num_ids:
        if id_end > num_ids:
            id_end = num_ids
        last_size += length_of_id_minishard(
            id_start, id_end, len(scalar_names))
        minishard_indexes[index] = last_size-(id_end-id_start)*24
        minishard_indexes[index + 1] = last_size
        id_start = id_end
        id_end += per_minishard
        index = index+2
    minishard_indexes[index:] = last_size+8
    np.asarray(minishard_indexes, dtype="<u8").tofile(f)

    id_start = 0
    id_end = per_minishard
    minishard_indexes = np.zeros(((2**minishard_bits) * 2))
    while id_start < num_ids:
        if id_end > num_ids:
            id_end = num_ids
        write_id_minishard(id_start, id_end, segments, f)
        id_start = id_end
        id_end += per_minishard



def write_id_shard_2(path, segments):
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "identity",
            "preshift_bits": 12,
            "minishard_bits": number_of_minishard_bits_ids(len(segments) - 1, 12),
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
