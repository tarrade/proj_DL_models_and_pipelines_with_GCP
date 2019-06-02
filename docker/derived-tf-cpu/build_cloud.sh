# get the poject ID
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=custom_container_image_conda
export IMAGE_TAG=dev-v1.0.13
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

echo $PROJECT_ID
echo $IMAGE_URI
echo $IMAGE_REPO_NAME
echo $IMAGE_TAG

cp ../../environment.yml environment.yml
cp ../../deployment/known_hosts known_hosts
cp ../../deployment/id_rsa id_rsa

gcloud builds submit --tag $IMAGE_URI . --timeout "2h00m0s"

rm environment.yml
rm id_rsa 
rm known_hosts