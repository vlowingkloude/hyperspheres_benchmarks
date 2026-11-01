#!/bin/bash

set -euo pipefail

DATASETS=(
    "gist|960|~/workspace/gist/|gist_|gist1m"
    "mrl_1024|1024|~/workspace/data/mrl_1024/||mrl_1024"
    "sift|128|~/workspace/sift/|sift_|sift1m"
)

SHUFFLE_DIR="$HOME/workspace/data/shuffles"
MAIN_EXE="$HOME/workspace/HyperspheresPruningRate/build/main/main"
REPEAT_COUNT=5

for ds_info in "${DATASETS[@]}"; do
    IFS='|' read -r name dim path prefix main_name <<< "$ds_info"
    
    path_expanded=$(eval echo "$path")
    
    echo "----------------------------------------------------------------"
    echo "Processing Dataset: $name (Dim: $dim)"
    echo "----------------------------------------------------------------"

    for ((i=1; i<=REPEAT_COUNT; i++)); do
        echo "Iteration $i/$REPEAT_COUNT for $name..."

        cp "${path_expanded}${prefix}base.fvecs" "$SHUFFLE_DIR/"
        cp "${path_expanded}${prefix}query.fvecs" "$SHUFFLE_DIR/"

        cd "$SHUFFLE_DIR" || exit
        
        ./shuffle "${prefix}base.fvecs" 1000000 "$dim" "./shu.fvecs"
        
        mv "./shu.fvecs" "${prefix}base.fvecs"
        
        ./gt "${prefix}base.fvecs" "${prefix}query.fvecs" "${prefix}groundtruth.ivecs"

        $MAIN_EXE \
          --engine hyperspheres --mode aggressive --dim "$dim" \
          --dataset "$main_name" --path "$SHUFFLE_DIR/" \
          --n_threads 4 --result $HOME/workspace/HyperspheresPruningRate/build/main/results --hs_n_ppage 0 --hs_beta 1.0

        RESULT_FILE="$HOME/workspace/HyperspheresPruningRate/build/main/results/hyperspheres-aggressive-1-0-${main_name}.json"
        NEW_FILE="$HOME/workspace/HyperspheresPruningRate/build/main/results/hyperspheres-aggressive-1-0-${main_name}-${i}.json"
        
        if [ -f "$RESULT_FILE" ]; then
            mv "$RESULT_FILE" "$NEW_FILE"
            echo "Saved result to $NEW_FILE"
        else
            echo "Error: Result file $RESULT_FILE not found!"
        fi
    done
done

echo "All experiments completed!"
