#!/usr/bin/env bash


# Trigger environment setup
./scripts/environment-setup.sh

# Fix npm issues
sudo chown -R 1000:1000 "/home/vscode/.npm"  

# Trigger prestart script
cd /workspaces/backend
source "$(poetry env info --path)/bin/activate"
./scripts/prestart.sh
deactivate
