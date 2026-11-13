#!/bin/bash

mkdir -p lrhnswbatch

RESULTP=lrhnswbatch
#SEARCHES=(100 200 300 400 500 600 700 800 900 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)
SEARCHES=(100 200 300 400 500 600 700 800 900 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2500 3000 3500 4000 4500)

for SEARCH in "${SEARCHES[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with search = $SEARCH"
    echo "--------------------------------------------------"

    ./build/spheres_benchmark \
  --engine faiss --mode hnsw --reuse_index 1 \
  --dataset gist1m --path ~/workspace/gist/ --distance squared_l2 \
  --n_threads 4 --metrics $RESULTP/gist_faiss_hnsw_$SEARCH.json \
  --index $RESULTP/gist_faiss_hnsw --faiss_hnsw_m 16 --faiss_hnsw_con 128 --faiss_hnsw_search $SEARCH

    if [ $? -eq 0 ]; then
        echo "Finished search $SEARCH successfully."
    else
        echo "Error occurred while running SEARCH $SEARCH"
        exit 1
    fi
done

echo "All experiments completed!"
