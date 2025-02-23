#!/bin/bash

set -e  # Exit on error

# Check if run ID is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a run ID"
    echo "Usage: $0 <run_id> [mode]"
    exit 1
fi

RUN_ID=$1
MODEL=${2:-scl}  # Use scl as default mode if not provided

echo "Using run: $RUN_ID"
echo "Using model: $MODEL   "

# Get threshold from training run
echo "Getting threshold from Task 2 training..."
THRESHOLD_T2=$(guild runs info "$RUN_ID" | grep "threshold:" | awk '{print $2}' | sed -n '2p')

if [ -z "$THRESHOLD_T2" ] || [ "$THRESHOLD_T2" = "null" ]; then
    echo "Error: Could not get threshold from training run for Task 2"
    exit 1
fi

echo "Threshold value for Task 2: $THRESHOLD_T2"

# Run test for Task 2
echo "Running tri-subject test for Task 2..."
guild run "$MODEL:tri_subject_test" \
    model.init_args.weights="exp/checkpoints/best.ckpt" \
    model.init_args.threshold="$THRESHOLD_T2" \
    "operation:$MODEL:tri_subject_train=$RUN_ID" -y \
    --gpus 0

echo "Task 2 completed successfully!" 
