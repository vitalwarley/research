#!/bin/bash

# Default GPU count is 7
gpu_count=${1:-7}

# Basic error handling for non-integer inputs
if ! [[ "$gpu_count" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU count must be an integer."
    exit 1
fi

# Loop for the specified number of GPUs
for ((i=0; i<gpu_count; i++)); do
  guild run --background --yes queue gpus=$i
done

