#!/bin/bash

# set python version, venv directory and requirements file
PYTHON_VERSION=3.10
VENV_NAME=venv
REQUIREMENTS=requirements.txt

# check if requirements file exists
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Requirements file not found!"
    exit 1
fi

# create a new venv if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    python$PYTHON_VERSION -m venv $VENV_NAME
fi

# activate the venv
source $VENV_NAME/bin/activate

# Update pip to the latest version
python -m pip install --upgrade pip

# update the requirements
pip3 install -r $REQUIREMENTS --upgrade

echo "Setup (or update) completed."