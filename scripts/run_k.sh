#!/bin/bash

rm -rf results/
mkdir results

KS=(1 10 20 30 40 50 60 70 80 90 100)

for K in "${KS[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with k = $K"
    echo "--------------------------------------------------"

    ./main \
      --engine hyperspheres \
      --mode aggressive \
      --dim 960 \
      --dataset "gist1m" \
      --path "$HOME/workspace/gist/" \
      --n_threads 4 \
      --result results \
      --hs_n_ppage 0 \
      --k $K \
      --hs_beta 1.0

    if [ $? -eq 0 ]; then
        mv results/hyperspheres-aggressive-1-0-gist1m.json results/hyperspheres-aggressive-1-0-$K-gist1m.json
        echo "Finished beta $K successfully."
    else
        echo "Error occurred while running beta $K"
        exit 1
    fi
done

echo "All experiments completed!"
