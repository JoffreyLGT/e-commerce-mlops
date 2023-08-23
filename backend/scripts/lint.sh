#!/usr/bin/env bash

# Get the full path to this script's directory and the path to backend dir
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir="$(dirname "$current_dir")"

# TODO Instead of settings folders here, set them in tool configs
folders="$backend_dir/app $backend_dir/scripts"

# Move into the folder so poetry can activate the environment
cd $backend_dir

# Count the amount of command that returned an error
# Each command will increment this value if they fail
ss=0

# Formatter
echo "----- Black -----"
poetry run black --check $folders && echo -e "Done.\n" || echo -e "Error.\n" && ((ss++))

# Linter
echo "----- Ruff -----"
if [[ $IS_GH_ACTION = "True" ]]; then
    poetry run ruff check --format=github $folders && echo -e "Done.\n" || echo -e "Error.\n" && ((ss++))
else
    poetry run ruff check $folders && echo -e "Done.\n" || echo -e "Error.\n" && ((ss++))
fi

# Type checker
echo "----- Mypy -----"
poetry run mypy $folders && echo -e "Done.\n" || echo -e "Error.\n" && ((ss++))

# Return the number of functions returning an error
exit $ss
