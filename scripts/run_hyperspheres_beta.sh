#!/bin/bash

rm -rf results/
mkdir results

BETAS=(-1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25)

for BETA in "${BETAS[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with beta = $BETA"
    echo "--------------------------------------------------"

    ./main \
      --engine hyperspheres \
      --mode aggressive \
      --dim 1024 \
      --dataset mrl_1024 \
      --path $HOME/workspace/data/mrl_1024 \
      --n_threads 4 \
      --result results \
      --hs_n_ppage 0 \
      --hs_beta $BETA

    if [ $? -eq 0 ]; then
        echo "Finished beta $BETA successfully."
    else
        echo "Error occurred while running beta $BETA"
        exit 1
    fi
done
