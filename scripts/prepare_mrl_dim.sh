#!/bin/bash

DIMS=(16 32 64 128 256 512 768)

for DIM in "${DIMS[@]}"
do
    echo "--------------------------------------------------"
    echo "Running with dim = $DIM"
    echo "--------------------------------------------------"

    mkdir mrl_$DIM

    ./extract mrl_1024/base.fvecs 1000000 1024 ./mrl_$DIM/base_raw.fvecs $DIM
    ./extract mrl_1024/query.fvecs 1000 1024 ./mrl_$DIM/query_raw.fvecs $DIM

    ./norm ./mrl_$DIM/base_raw.fvecs 1000000 $DIM ./mrl_$DIM/base.fvecs
    ./norm ./mrl_$DIM/query_raw.fvecs 1000 $DIM ./mrl_$DIM/query.fvecs

    rm -rf ./mrl_$DIM/base_raw.fvecs
    rm -rf ./mrl_$DIM/query_raw.fvecs

    ./gt ./mrl_$DIM/base.fvecs ./mrl_$DIM/query.fvecs ./mrl_$DIM/groundtruth.ivecs

    if [ $? -eq 0 ]; then
        echo "Finished dim $DIM successfully."
    else
        echo "Error occurred while running dim $DIM"
        exit 1
    fi
done

echo "All experiments completed!"
