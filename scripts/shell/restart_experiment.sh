#!/bin/bash

# Default to stopped experiments if no argument provided
status_flag="-Se"
status_type="stopped"
if [ "$1" = "pending" ]; then
    status_flag="-Sp"
    status_type="pending"
fi

echo "Starting experiment monitor loop..."
echo "Will check for $status_type experiments every 15 minutes"

while true; do
    # Check for running experiments
    running_check=$(guild select -Fo scl:train -Sr 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        # No running experiments found, proceed with restart
        echo "$(date): No running experiments found. Getting list of $status_type experiments..."
        
        # Get all experiments with specified status
        readarray -t exps < <(guild select -Fo scl:train $status_flag -A)
        
        if [ ${#exps[@]} -eq 0 ]; then
            echo "$(date): No $status_type experiments found to restart"
        else
            echo "$(date): Found ${#exps[@]} $status_type experiment(s)"
            
            # Limit to 6 experiments
            max_exps=6
            num_to_restart=$((${#exps[@]} > max_exps ? max_exps : ${#exps[@]}))
            
            echo "$(date): Will restart $num_to_restart $status_type experiment(s)"
            
            for ((i=0; i<num_to_restart; i++)); do
                exp=${exps[i]}
                echo "$(date): Restarting $status_type experiment on GPU $i: $exp"
                guild run --restart "$exp" -q --stage --gpus "$i" -y &
            done
            
            # Wait for all background processes
            wait
            echo "$(date): All $status_type experiment restart commands have been initiated"
        fi
    else
        echo "$(date): Found running experiment(s):"
        echo "$running_check"
    fi

    echo "$(date): Sleeping for 15 minutes before next check for $status_type experiments..."
    sleep 900  # 15 minutes in seconds
done 