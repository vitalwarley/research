#!/bin/bash

# Check if --dry-run flag is provided
dry_run=false
if [ "$1" == "--dry-run" ]; then
    dry_run=true
    echo "DRY RUN: No files will be deleted"
fi

# Get list of run directories from guild select
guild select -Fo "*:kinface-ft" -A | while read -r run_dir; do
    checkpoint_dir="$HOME/.guild/runs/$run_dir/exp/checkpoints"
    
    # Check if checkpoint directory exists
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Skipping: Checkpoint directory not found: $checkpoint_dir"
        continue
    fi

    echo "Processing: $checkpoint_dir"
    if [ "$dry_run" = true ]; then
        # Just list the files that would be deleted
        find "$checkpoint_dir" -name "*.ckpt" -type f -exec echo "Would delete: {}" \;
    else
        # Actually delete the files
        echo "Removing checkpoint files..."
        find "$checkpoint_dir" -name "*.ckpt" -type f -delete
    fi
done

echo "Process complete!" 