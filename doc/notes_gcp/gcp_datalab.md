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
  datalab list
  ```
## [Starting and Stopping an Datalab Instance](https://cloud.google.com/datalab/docs/how-to/lifecycle)
1. create your default instance
```
datalab create --machine-type n1-standard-2 instance-name --zone europe-west1-d
```
- see  `gcloud compute machine-types list` for all instances types in all zones
2. Stop you instance
  ```
  datalab stop instance-name
  ```
3. Reconnect (and restart) your instance
  ```
  datalab connect instance-name
  ```
> See API Reference: cloud.google.com/datalab/docs/reference-api
## Updating the VM type
If you have an instance running, you have to delete it before a more performant can be started. You will again be asked to specify you zone. Notebooks are stored separately in each zone.

```
datalab stop instance-name
datalab delete --keep-disk instance-name
```
`--keep-disk` is the default and can be omitted (you will be asked always if you want to proceed and if you keep or delete your persistent disk by proceeding)

Check where your persistent disk is located:

![Persistent Disk of Compute Engines](Figures/gcp_datalab_disks.png)

Then restart with more resources, e.g.:

```
datalab create --machine-type n1-standard-2 instance-name --zone zone-name
```

GPUs are only supported in beta-phase:

```
datalab beta create-gpu datalab-instance-name
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

## Delete your datalab instance 
- you need owner rights to do this
- you will always be asked if you want to proceed
  - without deleting the persistent disk
    ```
    datalab delete --keep-disk instance-name
    ```
  - deleting the persistent disk (make your notebooks are saved to repository!)
    ```
    datalab delete --delete-disk instance-name
    ```    