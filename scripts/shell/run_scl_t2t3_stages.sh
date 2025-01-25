#!/bin/bash

set -e  # Exit on error

# Check if run ID is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a run ID"
    echo "Usage: $0 <run_id>"
    exit 1
fi

RUN_ID=$1

# Get tau and alpha_neg from the provided run
echo "Getting tau and alpha_neg from run $RUN_ID..."
TAU=$(guild runs info "$RUN_ID" | grep "tau:" | awk '{print $2}')
ALPHA_NEG=$(guild runs info "$RUN_ID" | grep "alpha_neg:" | awk '{print $2}')

if [ -z "$TAU" ] || [ -z "$ALPHA_NEG" ]; then
    echo "Error: Could not get tau or alpha_neg from run"
    exit 1
fi

echo "Using tau: $TAU and alpha_neg: $ALPHA_NEG"

# Update both config files with the new tau and alpha_neg values
for config in "scl_t2.yml" "scl_t3.yml"; do
    echo "Updating $config..."
    sed -i "s/tau: .*/tau: $TAU/" "configs/$config"
    sed -i "s/alpha_neg: .*/alpha_neg: $ALPHA_NEG/" "configs/$config"
done

# Run Task 2 pipeline
echo "Running pipeline for Task 2..."

# Step 1: Training for Task 2
echo "Running tri-subject training..."
guild run "scl:tri_subject_train" -y

# Get the best run based on accuracy
BEST_RUN_T2=$(guild select -Fo scl:tri_subject_train --max accuracy)

if [ -z "$BEST_RUN_T2" ]; then
    echo "Error: No training runs found for Task 2"
    exit 1
fi

echo "Best run found for Task 2: $BEST_RUN_T2"

# Get threshold from training run
echo "Getting threshold from Task 2 training..."
THRESHOLD_T2=$(guild runs info "$BEST_RUN_T2" | grep "threshold:" | awk '{print $2}' | sed -n '2p')

if [ -z "$THRESHOLD_T2" ] || [ "$THRESHOLD_T2" = "null" ]; then
    echo "Error: Could not get threshold from training run for Task 2"
    exit 1
fi

echo "Threshold value for Task 2: $THRESHOLD_T2"

# Step 2: Run test for Task 2
echo "Running tri-subject test for Task 2..."
guild run scl:tri_subject_test \
    model.init_args.weights="exp/checkpoints/best.ckpt" \
    model.init_args.threshold="$THRESHOLD_T2" \
    operation:scl:tri_subject_train="$BEST_RUN_T2" -y

# Run Task 3 pipeline
echo "Running pipeline for Task 3..."

# Run search-retrieval using the model from RUN_ID
echo "Running search-retrieval with model from run $RUN_ID..."
guild run scl:search_retrieval operation:scl:train="$RUN_ID" -y

echo "Task 3 completed successfully!"

echo "All tasks completed successfully!" 
