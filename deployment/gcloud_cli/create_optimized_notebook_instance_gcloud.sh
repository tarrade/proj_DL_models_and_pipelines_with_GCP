# example AI Platform Notebook Instance using optimized image that derived from Deepp Learning image

# Name
export INSTANCE_NAME="test-setup-optimised-container-v01"

# Optimized images
export IMAGE_FAMILY="common-container"
export CONTAINER="gcr.io/PROJECT/xxxx"
#export CONTAINER="gcr.io/deeplearning-platform-release/tf-cpu:latest"

# Zone
export ZONE="europe-west6-a"
#export ZONE="europe-west1-b"

# Machine tyoe
export INSTANCE_TYPE="n1-standard-1"

gcloud compute instances create $INSTANCE_NAME \
--zone=$ZONE \
--subnet=projects/NETWORK/regions/europe-west6/subnetworks/SUBNET \
--image-family=$IMAGE_FAMILY \
--image-project=deeplearning-platform-release \
--machine-type=$INSTANCE_TYPE \
--boot-disk-size=80GB \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--metadata="proxy-mode=project_editors,container=${CONTAINER}" \
--labels application=notebook-familly-image,network=default,owner=name,type=pre-production,creation=gcloud