#! /usr/bin/env sh

# Script used to start the API.
# Warning: telemetry is not activated, no event will be sent to SigNoz.

# Get the full path to this script's directory and to the backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"

# Move into backend dir so poetry can activate environment
cd $backend_dir

# Exit in case of error
set -e

# Run prestart.sh to create DB
scripts/prestart.sh

echo "Start"
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
