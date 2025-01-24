#!/bin/bash

# Check if at least one run ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 run_id1 [run_id2 run_id3 ...]"
    echo "Example: $0 067cb42d99c1427fb90de18e028a4892 f6273180e73948419145a028b5a2d56b"
    exit 1
fi

# Loop through all provided run IDs
for run_id in "$@"; do
    # Create .guild directory if it doesn't exist
    mkdir -p "/home/warley/.guild/runs/$run_id/.guild"
    
    # Write the opref content
    echo "guildfile:/home/warley/dev/research/guild.yml $run_id scl train" > "/home/warley/.guild/runs/$run_id/.guild/opref"
done

echo "Fixed opref files for $# runs" 