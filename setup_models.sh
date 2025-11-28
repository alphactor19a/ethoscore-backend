#!/bin/bash
# Setup script to download models from GitHub
# This can be used as a build command in Render or run manually

set -e

echo "Installing git-lfs..."
# For Ubuntu/Debian (Render uses Ubuntu)
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y git-lfs
elif command -v brew &> /dev/null; then
    brew install git-lfs
else
    echo "Warning: Could not install git-lfs automatically. Please install it manually."
fi

# Initialize git-lfs
git lfs install || echo "Git LFS already initialized or not available"

echo "Setup complete. Models will be downloaded at application startup."

