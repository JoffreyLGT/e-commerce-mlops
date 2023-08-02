#! /usr/bin/env sh

# Script used to start the API with --reload option.
# Made to facilitate development.
# Warning: telemetry is not activated, no event will be sent to SigNoz.


# Run prestart.sh to create DB
./prestart.sh

echo "Start with reload"
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000