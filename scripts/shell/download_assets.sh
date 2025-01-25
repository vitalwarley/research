#!/bin/bash

# Check if data directory already exists
if [ ! -d "data" ]; then
    # Download dataset
    echo "Downloading dataset..."
    gdown 1e1hr_XKb5_IZxc5SXpjLeCJHrxrvXPWf

    # Extract dataset
    echo "Extracting dataset..."
    tar -xf data.tar.xz
    rm data.tar.xz  # Clean up compressed file
else
    echo "Data directory already exists, skipping dataset download..."
fi

# Create weights directory if it doesn't exist
mkdir -p weights

# Clone insightface if it doesn't exist
if [ ! -d "models/insightface" ]; then
    echo "Cloning insightface repository..."
    git clone https://github.com/deepinsight/insightface.git models/insightface
else
    echo "Insightface repository already exists, skipping clone..."
fi

# Check if weights file already exists
if [ ! -f "weights/adaface_ir101_webface12m.ckpt" ]; then
    # Download adaface model weights
    echo "Downloading adaface model weights..."
    gdown 1IqG6EOfOJVMgl9pFJDLCCfnFwYeZrJxB -O weights/adaface_ir101_webface12m.ckpt
else
    echo "Weights file already exists, skipping download..."
fi

# Check if weights file already exists
if [ ! -f "weights/ms1mv3_arcface_r100_fp16.pth" ]; then
    # Download arcface model weights
    echo "Downloading arcface model weights..."
    gdown 1SJnqCq-cbfsTsezkXOCmBTCD-0cQEVBn -O weights/ms1mv3_arcface_r100_fp16.pth
else
    echo "Weights file already exists, skipping download..."
fi

echo "Assets download check completed!" 
