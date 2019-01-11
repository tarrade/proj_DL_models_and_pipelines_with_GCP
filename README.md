# Develop DL models using Tensorflow and pipelines for industrialization and model serving

# General presentation and results of this project:
[Documentation](doc/DOC.md)

# Check if your Anaconda/python installation is configured "properly"
[Configuration](doc/SETUP.md)

# How to install the code

## Download the code from GitHub
- go to the directory in which you want to download the package from git  
- download the package from Github:   
  - ```git clone https://github.com/tarrade/proj_DL_models_and_pipelines_with_GCP.git```
  - or with other method from your choice (web interface, zip ...)   
- open an "Anaconda prompt" in the directory that contain the code from GitHub:   
  ```your_dir/proj_DL_models_and_pipelines_with_GCP/```

## Create the python conda env  
This will provide you a unique list of python packages needed to run the code

- create a python env based on a list of packages from environment.yml    
  ```conda env create -f environment.yml -n env_gcp_dl```
  
 - activate the env  
  ```conda activate env_gcp_dl```
  
 - in case of issue clean all the cache in conda
   ```conda clean -a -y```

## Update or delete the python conda env 
- update a python env based on a list of packages from environment.yml  
  ```conda env update -f environment.yml -n env_gcp_dl```

- delete the env to recreate it when too many changes are done  
  ```conda env remove -n env_gcp_dl```

## TensorBoard
- open a anaconda rompt
- go to the directory in which you want to download the package from git  
- activate the env:   
  ```conda activate env_gcp_dl```
- execute TensorBoard:   
  ```tensorboard --logdir ./results```
- open a web browser (Firefox):    
  ```http://C054093:6006```
  
 ![alt text](./doc/img/plot2.PNG)
 ![alt text](plot1.png)