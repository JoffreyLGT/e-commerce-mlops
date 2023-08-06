#!/usr/bin/env bash

# Print a trace of simple commands
set -x

# Formatter
black app scripts --check

# Linter
ruff app scripts --statistics

# Type checker
mypy app scripts