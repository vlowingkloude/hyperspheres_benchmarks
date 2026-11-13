#!/bin/bash


SEARCHES=(180)

for SEARCH in "${SEARCHES[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with search = $SEARCH"
    echo "--------------------------------------------------"

    ./build/spheres_benchmark \
  --engine faiss --mode ivf \
  --dataset gist1m --path ~/workspace/gist/ --distance squared_l2 \
  --n_threads 4 --metrics results/gist_faiss_ivf_$SEARCH.json \
  --index results/gist_faiss_ivf --faiss_ivf_nlists 4000 --faiss_ivf_nprobe $SEARCH

    if [ $? -eq 0 ]; then
        echo "Finished search $SEARCH successfully."
    else
        echo "Error occurred while running SEARCH $SEARCH"
        exit 1
    fi
done

echo "All experiments completed!"
