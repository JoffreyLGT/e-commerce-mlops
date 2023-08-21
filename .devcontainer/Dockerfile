FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

ENV PYTHONUNBUFFERED 1

# Execute shell commands as user vscode
USER vscode
# Store the user home path to be used in COPY instructions
ENV HOME /home/vscode

# Direnv configuration to reload .env variables when they change
COPY --chown=vscode .devcontainer/resources/direnv.toml $HOME/.config/direnv/

# ZSH Theme: p10k
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
COPY --chown=vscode .devcontainer/resources/.p10k.zsh $HOME/.p10k.zsh

# ZSH plugins
RUN git clone https://github.com/k-sriram/zsh-auto-venv ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-auto-venv --depth=1 \ 
&& git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
&& git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# ZSH config
COPY --chown=vscode .devcontainer/resources/.zshrc $HOME/.zshrc

# Poetry
RUN pipx install poetry
RUN mkdir -p ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/poetry \
    && poetry completions zsh > ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/poetry/_poetry

# Cache all packages
# The goal here is to cache all packages in the docker image, so setup is fast
ENV POETRY_VIRTUALENVS_IN_PROJECT true
# backend environment
COPY --chown=vscode backend/poetry.lock backend/pyproject.toml /workspaces/backend/
WORKDIR /workspaces/backend/
RUN poetry install -n

# backend environment
COPY --chown=vscode datascience/poetry.lock datascience/pyproject.toml /workspaces/datascience/
WORKDIR /workspaces/datascience/
RUN poetry install -n
