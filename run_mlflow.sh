#!/bin/bash

# this should be run from the ukrse2022_mlops_model_dev conda environment install from requirements_model_dev.yml

export UKRSE_DATA_ROOT=${HOME}/data/ukrse2022/
export PORT=5001
export MLFLOW_DB_PATH=sqlite:///${UKRSE_DATA_ROOT}/rse_mlops_mlflow.db
export MLFLOW_ARTIFACT_PATH=${UKRSE_DATA_ROOT}


echo database ${MLFLOW_DB_PATH}
echo port ${PORT}
mlflow server --port ${PORT} --backend-store-uri ${MLFLOW_DB_PATH}  --default-artifact-root ${MLFLOW_ARTIFACT_PATH} --host 0.0.0.0