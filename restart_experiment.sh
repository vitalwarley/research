#!/bin/bash

echo "Starting experiment monitor loop..."
echo "Will check for stopped experiments every 15 minutes"

while true; do
    # Check for running experiments
    running_check=$(guild select -Fo scl:train -Sr 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        # No running experiments found, proceed with restart
        echo "$(date): No running experiments found. Getting list of stopped experiments..."
        
        # Get all stopped experiments
        readarray -t stopped_exps < <(guild select -Fo scl:train -Se -A)
        
        if [ ${#stopped_exps[@]} -eq 0 ]; then
            echo "$(date): No experiments found to restart"
        else
            echo "$(date): Found ${#stopped_exps[@]} stopped experiment(s)"
            
            # Limit to 6 experiments
            max_exps=6
            num_to_restart=$((${#stopped_exps[@]} > max_exps ? max_exps : ${#stopped_exps[@]}))
            
            echo "$(date): Will restart $num_to_restart experiment(s)"
            
            for ((i=0; i<num_to_restart; i++)); do
                exp=${stopped_exps[i]}
                echo "$(date): Restarting experiment on GPU $i: $exp"
                guild run --restart "$exp" -q --stage --gpus "$i" -y &
            done
            
            # Wait for all background processes
            wait
            echo "$(date): All restart commands have been initiated"
        fi
    else
        echo "$(date): Found running experiment(s):"
        echo "$running_check"
    fi

    echo "$(date): Sleeping for 15 minutes before next check..."
    sleep 900  # 15 minutes in seconds
done 