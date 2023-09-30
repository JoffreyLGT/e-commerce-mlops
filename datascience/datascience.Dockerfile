FROM python:3 as base

ENV POETRY_VIRTUALENVS_IN_PROJECT true

RUN apt update \
    && apt upgrade -y \
    && apt install -y python3-dev \
    && apt install -y python3-venv \
    && apt install -y pipx \
    && apt install libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY datascience/deployment /mlruns
COPY datascience/pyproject.toml datascience/poetry.lock /datascience/
WORKDIR /datascience
RUN pip install poetry && poetry install --compile --no-interaction

COPY datascience /datascience
RUN chmod +x /datascience/scripts/start.sh

FROM base as development
CMD [ "scripts/start.sh" ]

FROM base as staging
CMD [ "scripts/start.sh" ]

FROM base as production
RUN poetry install --no-interaction --without dev --sync --compile
CMD [ "scripts/start.sh" ]
