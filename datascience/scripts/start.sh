#!/bin/bash

echo "Start MLFlow UI"
poetry run mlflow ui -h 0.0.0.0 -p $DATASCIENCE_MLFLOW_PORT \
--backend-store-uri "$MLFLOW_REGISTRY_URI" \
--default-artifact-root "$MLFLOW_TRACKING_URI" & \
echo "Start fusion model serving" & \
poetry run mlflow models serve -m "models:/fusion/Production" -h 0.0.0.0 -p $DATASCIENCE_MODEL_PORT --env-manager local

wait
