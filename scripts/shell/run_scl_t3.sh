#!/bin/bash

set -e  # Exit on error

# Check if run ID is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a run ID"
    echo "Usage: $0 <run_id> [model_name]"
    exit 1
fi

RUN_ID=$1
MODEL_NAME=${2:-scl}  # Use second argument if provided, default to 'scl'

# Run Task 3 pipeline
echo "Running pipeline for Task 3..."

# Run search-retrieval using the model from RUN_ID
echo "Running search-retrieval with model from run $RUN_ID..."
guild run ${MODEL_NAME}:search_retrieval operation:${MODEL_NAME}:train="$RUN_ID" model.init_args.weights=exp/checkpoints/best.ckpt -y

echo "Task 3 completed successfully!" 
