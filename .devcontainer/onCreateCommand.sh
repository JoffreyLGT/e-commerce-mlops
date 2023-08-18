#!/usr/bin/env bash


# Setup p10k
echo "Setup p10k"
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
cp -f .devcontainer/resources/.p10k.zsh ~/.p10k.zsh

# Setup autovenv
echo "Setup zsh-auto-venv"
git clone https://github.com/k-sriram/zsh-auto-venv ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-auto-venv --depth=1

# Copy zsh configs
echo "Setup Zsh"
cp -f .devcontainer/resources/.zshrc ~/.zshrc

# Trigger environment setup
./scripts/environment-setup.sh

# Trigger prestart script
source /workspaces/backend/.venv/bin/activate
cd /workspaces/backend
./scripts/prestart.sh
deactivate
