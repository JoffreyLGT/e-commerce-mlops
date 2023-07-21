#! /usr/bin/env sh

# Run prestart.sh to create DB
./prestart.sh

exec uvicorn app.main:app --reload