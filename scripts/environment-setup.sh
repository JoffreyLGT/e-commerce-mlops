#!/usr/bin/env bash

# Exit if an error occurs
set -e

# Get the full path to this script's directory
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
# And the parent dir to ensure we can call the scripts
root_dir="$(dirname "$current_dir")"
# Get current user name
user="$(id -un)"

echo "Install poetry-plugin-dotenv to load .env on run"
poetry self add poetry-plugin-dotenv

echo "Setup backend venv"
backend_venv=$root_dir/backend/.venv
# Check if backend_venv exists and if RESET_VENV is true
if [[ -d "$backend_venv" && "$RESET_VENV" == "true" ]]; then
    echo "RESET_VENV is set to true, delete backend venv to recreate it."
    rm -rf $backend_venv
elif [[ -d "$backend_venv" && ! -e "$backend_venv/$user" ]]; then
    echo "backend venv exists but was created by another user, delete it."
    rm -rf $backend_venv
fi
# Create the venv only if it doesn't exists.
if [ -d "$backend_venv" ]; then
    echo "Backend venv already exists."
else
    cd $root_dir/backend
    poetry install
    touch "$backend_venv/$user"
    echo "Done"
fi

echo "Setup datascience venv"
datascience_venv=$root_dir/datascience/.venv
# Check if datascience_venv exists and if RESET_VENV is true
if [[ -d "$datascience_venv" && "$RESET_VENV" == "true" ]]; then
    echo "RESET_VENV is set to true, delete datascience venv to recreate it."
    rm -rf $datascience_venv
elif [[ -d "$datascience_venv" && ! -e "$datascience_venv/$user" ]]; then
    echo "datascience venv exists but was created by another user, delete it."
    rm -rf $datascience_venv
fi

if [ -d "$datascience_venv" ]; then
    echo "datascience venv already exists."
else
    cd $root_dir/datascience
    poetry install
    touch "$datascience_venv/$user"
    echo "Done"
fi

echo "Remove existing Git hooks"
rm -v -f "$(git rev-parse --path-format=absolute --git-common-dir)"/hooks/*

echo "Install pre-commit hooks"
if command -v pre-commit &>/dev/null; then
    cd $root_dir
    pre-commit install
else
    echo "pre-commit command not found. Please install it and execute pre-commit install."
fi

# Start DB container for local environment
if [[ $IS_DEV_CONTAINER != "true" && $USE_DB_CONTAINER == "true" ]]; then
    cd $root_dir
    echo "Start db container"
    TARGET=development docker-compose -f docker-compose.yaml up -d db
    $root_dir/backend/scripts/prestart.sh
fi

echo "Environment successfully configured"
