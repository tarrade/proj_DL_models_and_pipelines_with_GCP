import os

from google.oauth2 import service_account
import googleapiclient.discovery

PROJECT_ID = ''
SERVICE_ACCOUNT_ID = ''

# # Get credentials
# credentials = service_account.Credentials.from_service_account_file(
#     filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
#     scopes=['https://www.googleapis.com/auth/cloud-platform'])

# Create the Cloud IAM service object
service = googleapiclient.discovery.build(
    'iam', 'v1')

# Call the Cloud IAM Roles API
# If using pylint, disable weak-typing warnings
# pylint: disable=no-member
body = "projects/{project_id}/serviceAccounts/{key_id}".format(project_id=PROJECT_ID, key_id=SERVICE_ACCOUNT_ID)

request = service.projects().serviceAccounts().undelete(name=body)
print(request.execute())
