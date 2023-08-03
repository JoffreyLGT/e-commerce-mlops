#! /usr/bin/env sh

# Exit in case of error
set -e

TARGET=test \
docker compose -f docker-compose.yaml config -o docker-stack.yaml

docker compose -f docker-stack.yaml build
docker compose -f docker-stack.yaml down -v --remove-orphans # Remove possibly previous broken stacks left hanging after an error
docker compose -f docker-stack.yaml up -d