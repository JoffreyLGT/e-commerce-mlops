#!/usr/bin/env bash

# Print a trace of simple commands
set -x

# Formatter
black app --check
# Import organiser
isort --check-only app
# Linter
pylint --rcfile pylintrc app
# docstring checker
pydocstyle