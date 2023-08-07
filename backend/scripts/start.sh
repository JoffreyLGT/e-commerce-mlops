#! /usr/bin/env sh

# Script used to start the API.
# Warning: telemetry is not activated, no event will be sent to SigNoz.

# Run prestart.sh to create DB
./scripts/prestart.sh

echo "Start"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000