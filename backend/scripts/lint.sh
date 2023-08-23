#!/usr/bin/env bash

# Get the full path to this script's directory
current_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
# And the parent dir to ensure we can call the scripts
backend_dir="$(dirname "$current_dir")"

cd $backend_dir

folders="$backend_dir/app $backend_dir/scripts"

# Count the amount of command that returned an error
# Each command will increment this value if they fail
ss=0

# Formatter
echo "----- Black -----"
black -q --check $folders && echo -e "Done.\n" || ((ss++))

# Linter
echo "----- Ruff -----"
if [[ $IS_GH_ACTION = "True" ]]
then
    ruff check --format=github $folders && echo -e "Done.\n"  || ((ss++))
else
    ruff check $folders && echo -e "Done.\n" || ((ss++))
fi

# Type checker
echo "----- Mypy -----"
mypy $folders && echo -e "Done.\n" || ((ss++))

exit $ss