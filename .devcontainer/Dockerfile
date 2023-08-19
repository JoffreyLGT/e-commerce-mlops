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
