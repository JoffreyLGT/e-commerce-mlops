#! /usr/bin/env sh

# Script used to start the API with telemetry.
# Made for non-dev environments.
# Warning: do not add --reload argument to uvicorn.
# See https://signoz.io/docs/instrumentation/fastapi/#steps-to-auto-instrument-fastapi-app-for-traces

# Get the full path to this script's directory and to the backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"
# Move into backend dir so poetry can activate environment
cd $backend_dir

# Exit in case of error
set -e

# Run prestart.sh to create DB
scripts/prestart.sh

echo "Start with telemetry"
poetry run opentelemetry-instrument --traces_exporter otlp_proto_http --metrics_exporter otlp_proto_http uvicorn app.main:app --host 0.0.0.0 --port 8000
