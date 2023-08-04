#!/usr/bin/env bash

# Print a trace of simple commands
set -x

# FIXME Add datascience folder

# Formatter
black app --check

# Linter
ruff app