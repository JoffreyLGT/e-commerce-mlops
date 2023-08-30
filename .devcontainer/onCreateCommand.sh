#!/usr/bin/env bash

# Trigger environment setup
./scripts/environment-setup.sh

# Trigger prestart script
cd /workspaces/backend
source "$(poetry env info --path)/bin/activate"
./scripts/prestart.sh
deactivate
