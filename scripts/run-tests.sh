#! /usr/bin/env sh


./scripts/tests/start-containers.sh

docker compose -f docker-stack.yaml exec -T api bash /backend/scripts/lint.sh
docker compose -f docker-stack.yaml exec -T api bash /backend/scripts/start-tests.sh "$@"

./scripts/tests/stop-containers.sh