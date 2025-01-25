#!/bin/bash

set -e  # Exit on error

# Step 1: Find best run based on accuracy
echo "Finding best run based on accuracy..."
BEST_RUN=$(guild select -Fo scl:train --max accuracy)

if [ -z "$BEST_RUN" ]; then
    echo "Error: No training runs found"
    exit 1
fi

echo "Best run found: $BEST_RUN"

# Step 2: Copy the config file directly from guild runs directory
CONFIG_PATH="$HOME/.guild/runs/$BEST_RUN/scl.yml"
if [ -f "$CONFIG_PATH" ]; then
    echo "Copying config file..."
    cp "$CONFIG_PATH" configs/scl.yml
else
    echo "Error: Config file not found in guild runs"
    exit 1
fi

# Check if validation run already exists
LATEST_VAL_RUN=$(guild select -Fo scl:val 2>/dev/null || echo "")
if [ -z "$LATEST_VAL_RUN" ]; then
    # Step 3: Run validation with best checkpoint
    echo "No validation run found, running validation..."
    guild run scl:val \
        model.init_args.weights="exp/checkpoints/best.ckpt" \
        operation:scl:train="$BEST_RUN" -y
    LATEST_VAL_RUN=$(guild select -Fo scl:val)
else
    echo "Validation run already exists, skipping to step 4..."
fi

# Step 4: Get the threshold from validation run
echo "Getting threshold from validation..."
if [ -z "$LATEST_VAL_RUN" ]; then
    echo "Error: No validation run found after running validation"
    exit 1
fi

# Not the precise value (it has 6 decimal places), but close enough
THRESHOLD=$(guild runs info "$LATEST_VAL_RUN" | grep "threshold:" | awk '{print $2}' | sed -n '2p')

if [ -z "$THRESHOLD" ] || [ "$THRESHOLD" = "null" ]; then
    echo "Error: Could not get threshold from validation run"
    exit 1
fi

echo "Threshold value: $THRESHOLD"

# Step 5: Run test with threshold
echo "Running test..."
guild run scl:test \
    model.init_args.weights="exp/checkpoints/best.ckpt" \
    model.init_args.threshold="$THRESHOLD" \
    operation:scl:train="$BEST_RUN" -y

echo "All stages completed successfully!" 
