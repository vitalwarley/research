#!/bin/bash

# Install Zellij if not already installed
if ! command -v zellij &> /dev/null; then
    echo "Installing Zellij..."
    # Get latest release URL
    LATEST_URL=$(curl -s https://api.github.com/repos/zellij-org/zellij/releases/latest | grep "browser_download_url.*x86_64-unknown-linux-musl.tar.gz" | cut -d '"' -f 4)
    # Download and extract
    curl -L "$LATEST_URL" -o zellij.tar.gz
    tar -xvf zellij.tar.gz
    # Move to local bin and make executable
    mkdir -p "$HOME/.local/bin"
    mv zellij "$HOME/.local/bin/"
    chmod +x "$HOME/.local/bin/zellij"
    # Clean up
    rm zellij.tar.gz
    echo "Zellij installed successfully!"
    # Add ~/.local/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    echo "Zellij is already installed!"
fi 
