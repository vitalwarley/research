#!/bin/bash

max_gpu=${1:-7}  # Use first argument if provided, default to 7 if not

for i in $(seq 0 $max_gpu)
do
  guild run --background --yes queue gpus=$i
done