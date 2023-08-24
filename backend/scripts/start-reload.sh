#! /usr/bin/env sh

# Script used to start the API with --reload option.
# Made to facilitate development.
# Warning: telemetry is not activated, no event will be sent to SigNoz.

# Stop execution if an error occurs
set -e

# Get the full path to this script's directory and the path to backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"

# Move into the folder so poetry can activate the environment
cd $backend_dir

# Run prestart.sh to create DB
scripts/prestart.sh

echo "Start with reload"
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
