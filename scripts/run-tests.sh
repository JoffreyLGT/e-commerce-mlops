#! /usr/bin/env sh

# Get the full path to this script's directory and to the backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
root_dir=$(dirname $current_dir)

cd $current_dir
tests/start-containers.sh

cd $current_dir
docker compose -f tests/docker-stack.yaml up -d
docker compose -f tests/docker-stack.yaml exec -T api bash poetry run python -m scripts.code_checking
docker compose -f tests/docker-stack.yaml exec -T api bash scripts/start-tests.sh

tests/stop-containers.sh
