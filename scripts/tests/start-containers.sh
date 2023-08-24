#! /usr/bin/env sh

# Exit in case of error
set -e

# Get the full path to this script's directory and to the root dir
script_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
root_dir=$(dirname $(dirname $script_dir))

cd $script_dir

TARGET=staging \
    docker compose -f $root_dir/docker-compose.yaml config -o docker-stack.yaml

docker compose -f docker-stack.yaml build
# Remove possibly previous broken stacks left hanging after an error
docker compose -f docker-stack.yaml down -v --remove-orphans
docker compose -f docker-stack.yaml up -d
