#!/bin/bash

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment and install dependencies in one go
uv venv --python=3.11 .venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -r pyproject.toml

echo "Setup completed successfully!" 
