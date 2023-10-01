#! /usr/bin/env bash

# Get the full path to this script's directory and to the backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"


# Exit in case of error
set -e

# Build and start containers
scripts/docker-deploy.sh --env-file staging.env

echo "Wait 5 seconds to give enough time for backend prestart script to be done"
sleep 5

# Execute start-tests script
docker exec --workdir /backend api-staging scripts/start-tests.sh
