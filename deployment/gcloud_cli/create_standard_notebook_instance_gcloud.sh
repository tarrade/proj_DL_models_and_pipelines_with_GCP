# example AI Platform Notebook Instance using standard images

# Name
export INSTANCE_NAME="test-setup-pytorch-v01"

# Tensorflow
#export IMAGE_FAMILY="tf-latest-cpu"
# PyTorch
export IMAGE_FAMILY="pytorch-latest-cpu"

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
--metadata="proxy-mode=project_editors" \
--labels application=notebook-familly-image,network=default,owner=name,type=pre-production,creation=gcloud