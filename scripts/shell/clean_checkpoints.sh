#!/bin/bash

# Initialize variables
dry_run=false
operation_pattern="*"  # Default to all operations
mode="all"  # Default to deleting all checkpoints

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
        --mode)
            if [ -n "$2" ]; then
                if [ "$2" = "all" ] || [ "$2" = "last" ]; then
                    mode="$2"
                    shift 2
                else
                    echo "Error: --mode must be either 'all' or 'last'"
                    exit 1
                fi
            else
                echo "Error: --mode requires a value (all/last)"
                exit 1
            fi
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--op OPERATION_PATTERN] [--mode all|last]"
            echo "Example: $0 --op '*:kinface-ft' --mode last"
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
        if [ "$mode" = "all" ]; then
            find "$checkpoint_dir" -name "*.ckpt" -type f -exec echo "Would delete: {}" \;
        else
            if [ -f "$checkpoint_dir/last.ckpt" ]; then
                echo "Would delete: $checkpoint_dir/last.ckpt"
            fi
        fi
    else
        # Actually delete the files
        echo "Removing checkpoint files..."
        if [ "$mode" = "all" ]; then
            find "$checkpoint_dir" -name "*.ckpt" -type f -delete
        else
            if [ -f "$checkpoint_dir/last.ckpt" ]; then
                rm "$checkpoint_dir/last.ckpt"
            fi
        fi
    fi
done

echo "Process complete!" 