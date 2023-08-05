#! /usr/bin/env sh


docker compose -f docker-stack.yaml down -v --remove-orphans
rm docker-stack.yaml