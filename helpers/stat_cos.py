import numpy as np
import os

def read_hs_file(file_path, dim, filesize, pagesize, page_index):
    page_start = pagesize * page_index
    id_offset = 16
    max_num_vectors_per_page = (pagesize - id_offset) // (8 + dim * 4)
    offset = id_offset + 8 * max_num_vectors_per_page
    with open(file_path, 'rb') as f:
        f.seek(page_start)
        n_bytes = f.read(8)
        if not n_bytes:
            raise EOFError("empty")
        
        n = np.frombuffer(n_bytes, dtype=np.int64)[0]
        f.seek(page_start + offset)
        data = np.fromfile(f, dtype=np.float32, count=n * dim)

        return data.reshape((n, dim))

def read_fvecs(filename, n, d):
    data = np.fromfile(filename, dtype=np.float32)
    data = data.reshape(n, d + 1)
    return data[:, 1:]

def get_orthogonal_component(sample, a, QM):
    diff = sample - a
    projection = QM @ (QM.T @ diff)
    orthogonal_v = diff - projection
    return orthogonal_v


hs_path = "$HOME/independent_runs_hyperspheres/hyperspheres_benchmarks/Hyperspheres/build/main/results/mrl_1024.hyperspheres"
query_path = "$HOME/workspace/data/mrl_1024/query.fvecs"
dim = 1024
filesize = 256*1024*1024
pagesize = 8*1024*1024
page_index = 20
n_queries = 1000
x = read_hs_file(hs_path + "/0", dim, filesize, pagesize, page_index)
q_all = read_fvecs(query_path, n_queries, dim)

x = x.astype(np.float64)
a = x[0].copy()
V = x - a
_, s, vh = np.linalg.svd(V, full_matrices = False)
random_dirs = np.random.default_rng(42).standard_normal((n_queries, dim))
random_dirs /= np.linalg.norm(random_dirs, axis = 1, keepdims = True)
original_norms = np.linalg.norm(V, axis = 1)
rtol = 1e-10
results = []
random_results = []
for mk in range(0, 16):
    M = dim * mk // 16
    QM = vh[:M].T
    V_perp = V - (V @ QM) @ QM.T
    norms = np.linalg.norm(V_perp, axis = 1)
    valid = norms > rtol * original_norms

    result = []
    random_result = []

    for i in range(n_queries):
        q = q_all[i, :]

        u = q.astype(np.float64) - a
        u_perp = u - QM @ (QM.T @ u)

        u_norm = np.linalg.norm(u_perp)

        coses = (V_perp[valid] / norms[valid, None]) @ (u_perp / u_norm)
        m = dim - M
        result.append([np.count_nonzero(~valid), np.mean(coses), np.var(coses), 1.0 / m, m * np.var(coses)])

        random_query = random_dirs[i, :]
        ru = random_query.astype(np.float64)
        ru_perp = ru - QM @ (QM.T @ ru)
        ru_norm = np.linalg.norm(ru_perp)
        rcoses = (V_perp[valid] / norms[valid, None]) @ (ru_perp / ru_norm)
        random_result.append([np.count_nonzero(~valid), np.mean(rcoses), np.var(rcoses), 1.0 / m, m * np.var(rcoses)])
    random_results.append(random_result)
    results.append(result)