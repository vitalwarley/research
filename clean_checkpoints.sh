#!/bin/bash

# Initialize variables
dry_run=false
operation_pattern="*"  # Default to all operations

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            dry_run=true
            echo "DRY RUN: No files will be deleted"
            shift
            ;;
        --op)
            if [ -n "$2" ]; then
                operation_pattern="$2"
                shift 2
            else
                echo "Error: --op requires an operation pattern"
                exit 1
            fi
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--op OPERATION_PATTERN]"
            echo "Example: $0 --op '*:kinface-ft'"
            exit 1
            ;;
    esac
done

# Get list of run directories from guild select
guild select -Fo "$operation_pattern" -A | while read -r run_dir; do
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