#!/usr/bin/env bash

# Print a trace of simple commands
set -x

# Formatter
black app scripts --check

# Linter
ruff app scripts

# Type checker
mypy app scripts