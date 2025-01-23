#!/bin/bash

max_gpu=${1:-4}     # First argument: max GPU number (default: 7)
gpu_per_run=${2:-1} # Second argument: GPUs per run (default: 1)

if [ "$gpu_per_run" != "1" ] && [ "$gpu_per_run" != "2" ]; then
    echo "Error: gpu_per_run must be 1 or 2"
    exit 1
fi

if [ "$gpu_per_run" = "1" ]; then
    # Single GPU mode
    for i in $(seq 0 $max_gpu)
    do
        guild run --background --yes queue gpus=$i
    done
else
    # Dual GPU mode
    for i in $(seq 0 $max_gpu)
    do
        next_gpu=$((i + 1))
        if [ $next_gpu -le $max_gpu ]; then
            guild run --background --yes queue gpus="$i,$next_gpu"
            i=$next_gpu  # Skip the next GPU since we've used it in this pair
        else
            # Handle the last GPU if we have an odd number
            guild run --background --yes queue gpus=$i
        fi
    done
fi
