#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch (Small size to prevent crash)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the libraries
pip install -r requirements.txt