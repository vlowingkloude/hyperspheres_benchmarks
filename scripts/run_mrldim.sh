#!/bin/bash

DIMS=(16 32 64 128 256 512 768 1024)

rm -rf results/
mkdir results

for DIM in "${DIMS[@]}"
do
    echo "--------------------------------------------------"
    echo "Running experiment with dim = $DIM"
    echo "--------------------------------------------------"
    SIZE=$((DIM * 8192))
    FILE_SIZE=$((SIZE * 32))

    ./main \
      --engine hyperspheres \
      --mode aggressive \
      --dim $DIM \
      --dataset "mrl_$DIM" \
      --path "$HOME/workspace/data/mrl_$DIM" \
      --n_threads 4 \
      --hs_page_size $SIZE \
      --hs_file_size $FILE_SIZE \
      --result results \
      --hs_n_ppage 0 \
      --hs_beta 1.0

      ./main \
      --engine hyperspheres \
      --mode safe \
      --dim $DIM \
      --dataset "mrl_$DIM" \
      --path "$HOME/workspace/data/mrl_$DIM" \
      --n_threads 4 \
      --hs_page_size $SIZE \
      --hs_file_size $FILE_SIZE \
      --result results \
      --hs_n_ppage 0

    if [ $? -eq 0 ]; then
        echo "Finished dim $DIM successfully."
    else
        echo "Error occurred while running dim $DIM"
        exit 1
    fi
done

echo "All experiments completed!"
