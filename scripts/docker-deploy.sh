#! /usr/bin/env bash

# Create all Docker containers necessary for the project,
# which includes the project containers and SigNoz containers.

# Stop if there is an error
set -e

# Store the script name for logging
me=$(basename "$0")

echo "$me - Setup datascience venv"
cd datascience
poetry install --only download && poetry run python -m scripts.download_mlruns
cd -

echo "$me - Using $2 as env file"
# Get variable from .env file
network=$(grep "DOCKER_NETWORK=" $2 | sed -e 's/.*=//')
echo "$me - Found DOCKER_NETWORK=$network"

echo "$me - Build images, create containers and start them detached"
docker compose $1 $2 -f docker-compose.yaml up --build --remove-orphans -d

if [[ "$TARGET_ENV" == "production" ]]; then
    cd scripts
    echo "$me - Run install-signoz script"
    ./install-signoz.sh -n $network
fi
