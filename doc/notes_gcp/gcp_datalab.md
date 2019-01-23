# Google Cloud Datalab

## [Working in Teams](https://cloud.google.com/datalab/docs/how-to/datalab-team)
-  [datalab instances](https://cloud.google.com/datalab/docs/how-to/datalab-team#create_instances_for_each_team_member) are single-user environments
   - instance name should contain username
   - owner can create user-instance
   ```
   datalab create datalab-<firstname>-<lastname> --for-user <firstname>.<lastname>@axa-winterthur.ch --zone europe-west1-d
   ```
   - user has to use `ungit` Tool to clone repository 
    1. start ungit over button

        ![ungit](https://cloud.google.com/datalab/images/ungit-icon.png)
    
    2. Then you start in the notebooks folder, where the *Cloud Source Repository* is situated

       ![ungit start view](https://cloud.google.com/datalab/images/ungit-open-repo.png)

    3. Change to `/content/datalab/` and clone the repositories you need.

        ![Clone Repos in unzip](Figures/unzip_clone_repo.png)
   - check IAM roles of user
- project owner has to create repository if a *Cloud Source Repository* should be created
  - option of `--no-create-repository` flag
  - list all created datalab instances, including their zones
  ```

  ```
## [Starting and Stopping an Datalab Instance](https://cloud.google.com/datalab/docs/how-to/lifecycle)
- create your default instance (here)
```
datalab create --machine-type n1-standard-2 instance-name --zone europe-west1-d
```
- see  `gcloud compute machine-types list` for all instances types in all zones


## Updating the VM type
```
datalab stop instance-name
datalab delete --keep-disk instance-name
datalab create --machine-type n1-standard-2 instance-name
```

> Datalab's persistent disk is strictly linked to a zone. Changing the zone recreates the persistent disk in the new zone without automatic file transfer

## Costs
- stopped datalab instances have cost for 
 - the external IP
 - the persistent disk

## [Updating the libraries](https://cloud.google.com/datalab/docs/how-to/adding-libraries)
]Besides the lightwight version using `!pip install lib-name`in a code cell of a notebook, it it possible to create a custom docker image specified in `Dockerfile-extended-example.in`
```
FROM datalab
...
pip install lib-name
...
```

## [Accessing Data](https://cloud.google.com/datalab/docs/how-to/working-with-notebooks#working_with_data)

