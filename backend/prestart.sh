#! /usr/bin/env bash

# Let the DB start
python app/scripts/backend_pre_start.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python app/scripts/seed_data.py

# Download model saves
python app/scripts/download_models.py