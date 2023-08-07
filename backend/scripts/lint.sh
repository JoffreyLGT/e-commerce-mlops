#!/usr/bin/env bash

# Get the full path to this script's directory
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
# And the parent dir to ensure we can call the scripts
backend_dir="$(dirname "$current_dir")"

cd $backend_dir

# Print a trace of simple commands
set -x

folders="$backend_dir/app $backend_dir/scripts"

# Formatter
black $folders --check

# Linter
if [[ $IS_GH_ACTION = "True" ]]
then
    ruff check --format=github $folders
else
    ruff check $folders
fi

# Type checker
mypy $folders
