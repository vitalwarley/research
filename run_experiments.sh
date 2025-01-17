#!/bin/bash

# Sampling configurations
# Format: "name,individual,family,relationship"
SAMPLING_CONFIGS=(
    "baseline,0,0,0"
    "balanced,0.33,0.33,0.34"
    "family_focused,0.2,0.6,0.2"
    "relationship_focused,0.6,0.2,0.2"
    "individual_focused,0.2,0.2,0.6"
)

# Parameter ranges
ALPHA_POS=(0.6 0.7 0.8)
ALPHA_NEG=(0.6 0.7 0.8)
TAU=(0.2 0.3 0.4)

# Loop through all combinations
for sampling in "${SAMPLING_CONFIGS[@]}"; do
    # Split sampling config into components
    IFS=',' read -r name ind fam rel <<< "$sampling"
    
    for ap in "${ALPHA_POS[@]}"; do
        for an in "${ALPHA_NEG[@]}"; do
            for t in "${TAU[@]}"; do
                # Run guild operation with all parameters
                guild run scl:train \
                    data.init_args.sampler=yes \
                    data.init_args.sampling_weights.ind="$ind" \
                    data.init_args.sampling_weights.fam="$fam" \
                    data.init_args.sampling_weights.rel="$rel" \
                    model.init_args.loss.init_args.alpha_pos="$ap" \
                    model.init_args.loss.init_args.alpha_neg="$an" \
                    model.init_args.loss.init_args.tau="$t" \
                    --stage-trials \
                    --quiet
            done
        done
    done
done 
