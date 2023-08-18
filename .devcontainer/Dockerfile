FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

ENV PYTHONUNBUFFERED 1

# Direnv configuration to reload .env variables when they change
RUN mkdir -p /home/vscode/.config/direnv
COPY .devcontainer/resources/direnv.toml /home/vscode/.config/direnv/
