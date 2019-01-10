# Develop DL models using Tensorflow and pipelines for industrialization and model serving

# Claims routing @ARAG

# General presentation and results of this project:
[Documentation](doc/DOC.md)

# Check if your Anaconda/python installation is configured "properly"
[Configuration](doc/SETUP.md)

# How to install the code

## Special configuration to download the code from GitHub with git

In order to be able to install packages from an external GitHub server, we need to change the .gitconfig file which you can find in `H:\.gitconfig`. Change the file to:

```
[user]
	name = Vorname Nachname
	email = mail@axa-winterthur.ch
[https]
    proxy = https://C-Nummer:My_Password@sc-wvs-ch-win-pr-01-vip1.ch.doleni.net:8080
    sslVerify = false
[http]
    proxy = http://C-Nummer:My_Password@sc-wvs-ch-win-pr-01-vip1.ch.doleni.net:8080
    sslVerify = false
[http "https://github.axa.com"]
    proxy =
    sslVerify = false
[https "https://github.axa.com"]
    proxy =
    sslVerify = false
[credential "https://github.axa.com"]
    username = mail@axa-winterthur.ch
[credential]
    helper = wincred
[core]
    autocrlf = true
    filemode = false
```

Same for pip, you need to create/change your `pip.ini` in *C:\Users\C-number\pip*

```
[list]
format=columns

[global]
trusted-host = github-production-release-asset-2e65be.s3.amazonaws.com 
proxy = 'https://C-Nummer:My_Password@sc-wvs-ch-win-pr-01-vip1.ch.doleni.net:8080
```

Unfortunately, you need to type in your Windows password. Also if your password contains special characters you need to encode them according to https://www.w3schools.com/tags/ref_urlencode.asp.

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
  ```activate env_gcp_dl```
  
 - in case of issue clean all the cache in conda
   ```conda clean -a -y```

## Update or delete the python conda env 
- update a python env based on a list of packages from environment.yml  
  ```conda env update -f environment.yml -n env_gcp_dl```

- delete the env to recreate it when too many changes are done  
  ```conda env remove -n env_gcp_dl```

