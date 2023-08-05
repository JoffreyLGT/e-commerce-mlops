#! /usr/bin/env bash

# Let the DB start
python scripts/backend_pre_start.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python scripts/seed_data.py

# Download model saves
python scripts/download_models.py