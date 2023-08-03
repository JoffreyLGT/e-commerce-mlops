FROM python:3 as base

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=.

COPY ./requirements.txt /backend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt
RUN rm /backend/requirements.txt
COPY ./ /backend
WORKDIR /backend

FROM base as dev
COPY ./requirements-dev.txt /backend/requirements-dev.txt
RUN pip install --no-cache-dir --upgrade -r /backend/requirements-dev.txt
RUN rm /backend/requirements-dev.txt
RUN chmod +x /backend/scripts/start-reload.sh
CMD [ "./scripts/start-reload.sh" ]

FROM base as test
COPY ./requirements-dev.txt /backend/requirements-dev.txt
RUN pip install --no-cache-dir --upgrade -r /backend/requirements-dev.txt
RUN rm /backend/requirements-dev.txt
RUN chmod +x /backend/scripts/start-tests.sh
CMD [ "./scripts/start-tests.sh" ]

FROM base as prod
RUN chmod +x /backend/scripts/start-with-telemetry.sh
CMD [ "./scripts/start-with-telemetry.sh" ]

