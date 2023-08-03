#! /usr/bin/env sh

# Store the script name for logging purpose
me=$(basename "$0")

# Exit in case of error
set -e
# Print a trace of simple commands
set -x

# Run prestart.sh to create DB
./scripts/prestart.sh

pytest app/tests "${@}"