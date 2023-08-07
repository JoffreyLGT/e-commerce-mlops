#! /usr/bin/env bash

# Get the full path to this script's directory
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
# And the parent dir to ensure we can call the scripts
backend_dir="$(dirname "$current_dir")"

# Exit in case of error
set -e

# Run prestart.sh to create DB
PYTHONPATH=$backend_dir $current_dir/prestart.sh

PYTHONPATH=$backend_dir pytest $backend_dir/app/tests --disable-warnings -q
