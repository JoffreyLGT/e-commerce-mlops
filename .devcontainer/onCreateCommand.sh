#!/usr/bin/env bash


# Trigger environment setup
./scripts/environment-setup.sh

# Fix npm issues
sudo chown -R 1000:1000 "/home/vscode/.npm"  

# Trigger prestart script
source /workspaces/backend/.venv/bin/activate
cd /workspaces/backend
./scripts/prestart.sh
deactivate
