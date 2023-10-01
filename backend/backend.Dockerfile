FROM python:3 as base

ENV PYTHONUNBUFFERED 1
ENV POETRY_VIRTUALENVS_IN_PROJECT true
ENV RESET_VENV true

RUN apt update \
    && apt upgrade -y \
    && apt install -y python3-dev \
    && apt install -y python3-venv \
    && apt install -y pipx \
    && apt install libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pipx ensurepath \
    && pipx install poetry
# Make pipx packages accessible
ENV PATH=/root/.local/bin:$PATH

COPY backend/pyproject.toml backend/poetry.lock /backend/
WORKDIR /backend

FROM base as development
RUN poetry install -n
COPY ./backend/ /backend/
COPY mypy.ini /backend/
RUN chmod +x /backend/scripts/start-reload.sh
CMD [ "./scripts/start-reload.sh" ]

FROM base as staging
RUN poetry install -n
COPY ./backend/ /backend/
COPY mypy.ini /backend/
RUN chmod +x /backend/scripts/start.sh
CMD [ "./scripts/start.sh" ]

FROM base as production
RUN poetry install -n --without dev
COPY ./backend/ ../mypy.ini /backend/
COPY mypy.ini /backend/
RUN chmod +x /backend/scripts/start-with-telemetry.sh
CMD [ "./scripts/start-with-telemetry.sh" ]
