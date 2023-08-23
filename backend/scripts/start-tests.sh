#! /usr/bin/env bash

# Get the full path to this script's directory and to the backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"

# Move into backend dir so poetry can activate environment
cd $backend_dir

# Exit in case of error
set -e

# Run prestart.sh to create DB
scripts/prestart.sh
# Start tests
poetry run pytest $backend_dir/app/tests --disable-warnings -q
