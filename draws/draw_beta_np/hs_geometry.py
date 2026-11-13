from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_HS = Path(
    "$HOME/independent_runs_hyperspheres/"
    "hyperspheres_benchmarks/Hyperspheres/build/main/results/"
    "mrl_1024.hyperspheres/10"
)
DEFAULT_QUERY = Path("$HOME/workspace/data/mrl_1024/query.fvecs")
DEFAULT_CACHE = HERE / "hs_geometry.npz"

DIM = 1024
PAGE_SIZE = 8 * 1024 * 1024
PAGE_INDEX = 30
N_QUERIES = 1000
N_M = 16


def read_hs_file(file_path: Path, dim: int, pagesize: int,
                 page_index: int) -> np.ndarray:
    """Read the vectors stored in one page of a .hyperspheres file."""
    page_start = pagesize * page_index
    id_offset = 16
    max_num_vectors = (pagesize - id_offset) // (8 + dim * 4)
    offset = id_offset + 8 * max_num_vectors
    with open(file_path, "rb") as fh:
        fh.seek(page_start)
        n_bytes = fh.read(8)
        if not n_bytes:
            raise EOFError(f"empty page {page_index} in {file_path}")
        n = int(np.frombuffer(n_bytes, dtype=np.int64)[0])
        fh.seek(page_start + offset)
        data = np.fromfile(fh, dtype=np.float32, count=n * dim)
    return data.reshape((n, dim))


def read_fvecs(filename: Path, n: int, d: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float32)
    data = data.reshape(n, d + 1)
    return data[:, 1:]


def compute(hs_path: Path = DEFAULT_HS, query_path: Path = DEFAULT_QUERY,
            n_queries: int = N_QUERIES, dim: int = DIM,
            page_index: int = PAGE_INDEX) -> dict:
    """Return per-query mean/var/scaled-moment arrays, shape (16, n_queries, 3)."""
    x = read_hs_file(hs_path, dim, PAGE_SIZE, page_index).astype(np.float64)
    q_all = read_fvecs(query_path, n_queries, dim).astype(np.float64)

    anchor = x[0].copy()
    V = x - anchor
    _, _, vh = np.linalg.svd(V, full_matrices=False)

    rng = np.random.default_rng(42)
    random_dirs = rng.standard_normal((n_queries, dim))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    original_norms = np.linalg.norm(V, axis=1)
    rtol = 1e-10
    real = np.empty((N_M, n_queries, 3), dtype=np.float64)
    random = np.empty_like(real)

    for mk in range(N_M):
        M = dim * mk // N_M
        QM = vh[:M].T
        V_perp = V - (V @ QM) @ QM.T
        norms = np.linalg.norm(V_perp, axis=1)
        valid = norms > rtol * original_norms
        # Hoisted out of the query loop: the original expression recomputed this
        # 2044 x 1024 division for every one of the 1000 queries.
        Vn = V_perp[valid] / norms[valid, None]
        m = dim - M

        for i in range(n_queries):
            u = q_all[i] - anchor
            u_perp = u - QM @ (QM.T @ u)
            u_norm = np.linalg.norm(u_perp)
            coses = Vn @ (u_perp / u_norm)
            mean = float(np.mean(coses))
            var = float(np.var(coses))
            real[mk, i] = (mean, var, m * (var + mean * mean))

            ru = random_dirs[i]
            ru_perp = ru - QM @ (QM.T @ ru)
            ru_norm = np.linalg.norm(ru_perp)
            rcoses = Vn @ (ru_perp / ru_norm)
            rmean = float(np.mean(rcoses))
            rvar = float(np.var(rcoses))
            random[mk, i] = (rmean, rvar, m * (rvar + rmean * rmean))

    return {
        "x": np.arange(N_M) / N_M,
        "real": real,
        "random": random,
        "dim": dim,
        "n_queries": n_queries,
        "page_index": page_index,
    }


def load(cache: Path = DEFAULT_CACHE, recompute: bool = False,
         hs_path: Path = DEFAULT_HS, query_path: Path = DEFAULT_QUERY,
         n_queries: int = N_QUERIES, dim: int = DIM,
         page_index: int = PAGE_INDEX) -> dict:
    """Load the cached statistics, computing them first when needed."""
    if cache.is_file() and not recompute:
        with np.load(cache) as z:
            return {key: z[key] for key in z.files}
    print(f"computing cosine statistics from {hs_path} ...")
    data = compute(hs_path, query_path, n_queries, dim, page_index)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"wrote {cache}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs-path", type=Path, default=DEFAULT_HS)
    ap.add_argument("--query-path", type=Path, default=DEFAULT_QUERY)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--page-index", type=int, default=PAGE_INDEX)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    data = load(args.cache, args.recompute, args.hs_path, args.query_path,
                args.n_queries, args.dim, args.page_index)
    print(f"x: {data['x'][0]:.4f}..{data['x'][-1]:.4f} ({len(data['x'])} points)")
    print(f"real mean cos: {data['real'][:, :, 0].mean(axis=1)}")
    print(f"random mean cos: {data['random'][:, :, 0].mean(axis=1)}")


if __name__ == "__main__":
    main()
