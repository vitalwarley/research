#!/bin/bash

set -e  # Exit on error

# Check if run ID is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a run ID"
    echo "Usage: $0 <run_id>"
    exit 1
fi

RUN_ID=$1

# Run Task 3 pipeline
echo "Running pipeline for Task 3..."

# Run search-retrieval using the model from RUN_ID
echo "Running search-retrieval with model from run $RUN_ID..."
guild run scl:search_retrieval operation:scl:train="$RUN_ID" -y

echo "Task 3 completed successfully!" 
