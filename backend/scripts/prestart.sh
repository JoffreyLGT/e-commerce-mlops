#! /usr/bin/env bash

# Prestart script ensuring the API is reading to be started.

# Get script and backend directory
script_dir=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
backend_dir=$(dirname $script_dir)

# Ensure the execution is from the backend dir
cd $backend_dir

# Let the DB start
poetry run python scripts/backend_pre_start.py

# Run migrations
poetry run alembic upgrade head

# Create initial data in DB
poetry run python scripts/seed_data.py

# Download model saves
poetry run python scripts/download_models.py
