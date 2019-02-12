#!/bin/bash
IMAGE=gcr.io/cloud-datalab/datalab:latest
CONTENT=/Users/tarrade/Desktop/Work/Data_Science/Tutorials_Codes/Python/Jupyter_Notebook/proj_DL_models_and_pipelines_with_GCP # your git repo
PROJECT_ID=$(gcloud config get-value project)

if [ "$OSTYPE" == "linux"* ]; then
  PORTMAP="127.0.0.1:8081:8080"
else
  PORTMAP="8081:8080"
fi


docker pull $IMAGE
docker run -it \
   -p $PORTMAP \
   -v "$CONTENT:/content" \
   -e "PROJECT_ID=$PROJECT_ID" \
   $IMAGE