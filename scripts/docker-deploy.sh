#! /usr/bin/env bash

# Create all Docker containers necessary for the project,
# which includes the project containers and SigNoz containers.

# Stop if there is an error
set -e

# Check for the build
if [[ -z "$ENV_TARGET" ]]; then
    echo "Run in 'development' build. To target another build, you must provide a environment variable called ENV_TARGET with the value 'development', 'staging' or 'production'."
    export TARGET="development"
else
    echo "Run in $ENV_TARGET"
    export TARGET=$ENV_TARGET
fi

# Get variable from .env file
network=$(grep "DOCKER_NETWORK=" .env | sed -e 's/.*=//')

# Store the script name for logging
me=$(basename "$0")
# Ensure we are in the script directory
reldir="$(dirname -- "$0")"
cd "$reldir"

echo "$me - Create project containers without starting them"
docker compose -f "../docker-compose.yaml" create --build --force-recreate

if [[ "$ENV_TARGET" == "production" ]]; then
    echo "$me - Run install-signoz script"
    ./install-signoz.sh -n $network
fi

if [[ $1 == "up" ]]; then
    docker compose -f ../docker-compose.yaml "$@"
fi
