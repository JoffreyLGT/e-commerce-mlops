#!/usr/bin/env bash

# Print a trace of simple commands
set -x

folders="app scripts"

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
