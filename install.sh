#!/bin/bash

EDITABLE_MODE=$1  # Pass "editable" as the first argument for editable mode

# Install dependencies

pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
    echo "Error installing dependencies"
    exit 1
fi

# Install magi_to_hstu cuda package

cd csrc/utils/magi_to_hstu

if [ "$EDITABLE_MODE" == "editable" ]; then
    echo "Installing magi_to_hstu in editable mode..."
    pip install -e . --no-build-isolation
else
    echo "Installing magi_to_hstu in standard mode..."
    pip install . --no-build-isolation
fi

if [[ $? -ne 0 ]]; then
    echo "Error installing magi_to_hstu package"
    exit 1
fi

# Install create_block_mask cuda package

cd -
cd csrc/utils/create_block_mask

if [ "$EDITABLE_MODE" == "editable" ]; then
    echo "Installing create_block_mask in editable mode..."
    pip install -e . --no-build-isolation
else
    echo "Installing create_block_mask in standard mode..."
    pip install . --no-build-isolation
fi

if [[ $? -ne 0 ]]; then
    echo "Error installing create_block_mask package"
    exit 1
fi

# Install flash_attn_cute package

cd -
cd flash_attn/

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
