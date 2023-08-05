#! /usr/bin/env bash

# Create all Docker containers necessary for the project,
# which includes the project containers and SigNoz containers.

# Get variable from .env file
network=$(grep "DOCKER_NETWORK=" .env | sed -e 's/.*=//')
target=$(grep "ENV_TARGET=" .env | sed -e 's/.*=//')
# Store the script name for logging
me=$(basename "$0")
# Ensure we are in the script directory
reldir="$( dirname -- "$0"; )";
cd "$reldir";

echo "$me - Create project containers without starting them"
export TARGET=$target
docker compose -f "../docker-compose.yaml" create --build --force-recreate

echo "$me - Run install-signoz script"
./install-signoz.sh -n $network
