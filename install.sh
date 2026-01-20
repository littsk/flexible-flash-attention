#!/bin/bash

EDITABLE_MODE=$1  # Pass "editable" as the first argument for editable mode

# Install dependencies

pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
    echo "Error installing dependencies"
    exit 1
fi

# Step 1: Build create_block_mask using Makefile

cd csrc/utils

echo "=============================================="
echo "Building create_block_mask CUDA extension..."
echo "=============================================="
make block_mask

if [[ $? -ne 0 ]]; then
    echo "Error building create_block_mask package"
    exit 1
fi

# Step 2: Install flash_attn_cute package

cd -
cd flash_attn/cute

if [ "$EDITABLE_MODE" == "editable" ]; then
    echo "Installing flash_attn_cute in editable mode..."
    pip install -e . --no-build-isolation
else
    echo "Installing flash_attn_cute in standard mode..."
    pip install . --no-build-isolation
fi

if [[ $? -ne 0 ]]; then
    echo "Error installing flash_attn_cute package"
    exit 1
else
    echo "flash_attn_cute package installed successfully !"
fi
