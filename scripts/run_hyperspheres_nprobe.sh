#!/bin/bash

rm -rf results/
mkdir results

PAGES=(25 75 125 175 225 275 325 375 425 475 490)

for PAGE in "${PAGES[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with page = $PAGE"
    echo "--------------------------------------------------"

    ./main \
      --engine hyperspheres \
      --mode aggressive \
      --dim 1024 \
      --dataset "mrl_1024" \
      --path "$HOME/workspace/data/mrl_1024" \
      --n_threads 4 \
      --result results \
      --hs_n_ppage $PAGE \
      --hs_beta 1.0

    if [ $? -eq 0 ]; then
        echo "Finished beta $PAGE successfully."
    else
        echo "Error occurred while running beta $PAGE"
        exit 1
    fi
done

echo "All experiments completed!"
