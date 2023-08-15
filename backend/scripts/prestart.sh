#! /usr/bin/env bash

dir_fullpath=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))

# Let the DB start
python $dir_fullpath/backend_pre_start.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python $dir_fullpath/seed_data.py

# Download model saves
python $dir_fullpath/download_models.py
