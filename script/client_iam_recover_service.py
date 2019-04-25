import os

import googleapiclient.discovery

#replace with correct information:
PROJECT_ID = 'ml-productive-pipeline-12345'
SERVICE_ACCOUNT_UNIQUE_ID = '111111111111111111111'
SERVICE_ACCOUNT_EMAIL_ID = '123456789123-compute@developer.gserviceaccount.com'

# Create the Cloud IAM service object
service = googleapiclient.discovery.build('iam', 'v1') # credentials from Cloud SDK

# Call the Cloud IAM Roles API

# Did not work for now
body = "projects/{project_id}/serviceAccounts/{key_id}/".format(
    project_id=PROJECT_ID, 
    key_id=SERVICE_ACCOUNT_UNIQUE_ID)

request = service.projects().serviceAccounts().undelete(name=body)
print(request.execute())

body = "projects/{project_id}/serviceAccounts/{key_id}".format(
    project_id=PROJECT_ID, 
    key_id=SERVICE_ACCOUNT_EMAIL_ID)

request = service.projects().serviceAccounts().undelete(name=body)
print(request.execute())