To build:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

We used openblas

Example usage:

```
./build/spheres_benchmark \
  --engine faiss --mode hnsw \
  --dataset gist1m --path /data/gist --distance squared_l2 \
  --n_threads 4 --metrics results/gist_faiss_hnsw.json \
  --index results/ --faiss_hnsw_m 16 --faiss_hnsw_con 128 --faiss_hnsw_search 200
```
