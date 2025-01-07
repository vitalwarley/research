#!/bin/bash

# Default values
METRIC="auc"
OPERATION="kinface-ft"
RUN_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --operation)
            OPERATION="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate operation
if [[ ! "$OPERATION" =~ ^(kinface-ft|kinface-ce)$ ]]; then
    echo "Error: operation must be either 'kinface-ft' or 'kinface-ce'"
    exit 1
fi

# Get best run based on run-id or metric
if [ -n "$RUN_ID" ]; then
    echo "Finding run matching ID substring: $RUN_ID..."
    BEST_RUN=$(guild select "$RUN_ID")
else
    # Validate metric if being used
    if [[ ! "$METRIC" =~ ^(auc|accuracy)$ ]]; then
        echo "Error: metric must be either 'auc' or 'accuracy'"
        exit 1
    fi
    echo "Exporting best model based on $METRIC..."
    BEST_RUN=$(guild select -Fo scl:train --max "$METRIC" -n1)
fi

if [ -z "$BEST_RUN" ]; then
    echo "Error: No matching runs found"
    exit 1
fi

echo "Best run found: $BEST_RUN"

# Check if run is already exported
if [ -d "weights/$BEST_RUN" ]; then
    echo "Run already exported to weights directory, skipping export..."
else
    # Export the run to weights/
    echo "Exporting run to weights directory..."
    guild export weights "$BEST_RUN" -y
fi

# Find the checkpoint file
CKPT_PATH=$(find "weights/$BEST_RUN/exp/checkpoints" -name "*.ckpt" -not -name "last.ckpt")

if [ -z "$CKPT_PATH" ]; then
    echo "Error: No checkpoint file found"
    exit 1
fi

echo "Found checkpoint: $CKPT_PATH"

# Run the specified KinFace operation
echo "Running $OPERATION..."
if [ "$OPERATION" = "kinface-ft" ]; then
    # Run for both datasets and all folds
    guild run scl:$OPERATION \
        model.init_args.weights="$CKPT_PATH" \
        data.init_args.dataset=[I,II] \
        data.init_args.fold=[1,2,3,4,5] -y
else
    guild run scl:$OPERATION -y
fi 