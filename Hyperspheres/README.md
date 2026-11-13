Hyperspheres with some patches to report pruning rate, and some helper to benchmark 

To build:

```sh
mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -S .. -B .

cmake --build . --target main

```


CLI Args:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--dataset` | `string` | name of the dataset: gist1m, sift1m, or mrl_dim |
| `--engine` | `string` | hyperspheres |
| `--mode` | `string` | safe or aggressive |
| `--path` | `string` | path to the dataset files |
| `--result` | `string` | output path |
| `--n_threads` | `integer` | number of threads. |
| `--dim` | `integer` | dimensionality of the vectors |
| `--n_base` | `integer` | number of base vectors |
| `--n_learn` | `integer` | number of learning vectors |
| `--n_query` | `integer` | number of query vectors |
| `--k` | `integer` | $k$ for neighbor search |
| `--kgt` | `integer` | $k$ in ground truth |
| `--hs_page_size` | `integer` | page size, 8MB by default |
| `--hs_file_size` | `integer` | file size, 256MB by default |
| `--hs_n_ppage` | `integer` | nprobe, number of pages to probe |
| `--hs_buf_size` | `integer` | thread-local buffer size |
| `--hs_morsel` | `integer` | morsel step size |
| `--hs_beta` | `float` | beta |


For example

```sh
./main \
  --engine hyperspheres --mode aggressive --dim 1024 \
  --dataset mrl_1024 --path ~/workspace/data/mrl_1024 \
  --n_threads 4 --result results --hs_n_ppage 0 --hs_beta 1.0
```
