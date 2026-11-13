#!/bin/bash

rm -rf results/
mkdir results

BETAS=(0.9)
PAGES=(10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 490)

for BETA in "${BETAS[@]}"
do
  for PAGE in "${PAGES[@]}"
  do
    echo "--------------------------------------------------"
    echo "Running experiment with beta = $BETA" page = $PAGE
    echo "--------------------------------------------------"

    ./main \
      --engine hyperspheres \
      --mode aggressive \
      --dim 960 \
      --dataset gist1m \
      --path $HOME/workspace/gist \
      --n_threads 4 \
      --result results \
      --hs_n_ppage $PAGE \
      --hs_beta $BETA

    if [ $? -eq 0 ]; then
        echo "Finished beta $BETA successfully."
    else
        echo "Error occurred while running beta $BETA"
        exit 1
    fi
  done
done

echo "All experiments completed!"
