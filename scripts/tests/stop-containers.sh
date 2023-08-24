#! /usr/bin/env sh

# Get the full path to this script's directory and to the backend dir
script_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
cd $script_dir

docker compose -f docker-stack.yaml down -v --remove-orphans
rm docker-stack.yaml
