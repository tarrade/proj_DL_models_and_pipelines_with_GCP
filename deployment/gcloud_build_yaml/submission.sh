# deploy an optimized notebok instance
#gcloud deployment-manager deployments create deployment-notebook-v01 --config create_optimized_notebook_instance.yaml

# deploy an standard notebok instance
gcloud deployment-manager deployments create deployment-notebook-sandor-v01 --config create_standard_notebook_instance.yaml

# deploy a docker VM
#gcloud deployment-manager deployments create deployment-debian-v01 --config create_docker_vm.yaml
