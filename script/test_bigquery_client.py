"""
running example with proxy.
authentification using skd credentials still does not work. (Maybe due to proxy?)
"""
# pip install --upgrade google-cloud-bigquery
from google.cloud import bigquery
import requests

from config_client import proxy_https
# from config_client import filepath_credentials
from config_client import filepath_axa_cert
from config_client import PROJECT_ID
from google.auth.transport.requests import AuthorizedSession

import os

#should work: https://google-auth.readthedocs.io/en/latest/user-guide.html#application-default
import google.auth
cred, project = google.auth.default()


# s = requests.Session()
s = AuthorizedSession(cred)
# s.verify=False # also possible, not using the ssl-certificate
s.verify = filepath_axa_cert
s.proxies = {
    "https": proxy_https,
}

# # https://google-auth.readthedocs.io/en/latest/user-guide.html#service-account-private-key-files

# # ## 1.
# from oauth2client.client import GoogleCredentials
# cred = GoogleCredentials.get_application_default()



# ## 2.
# from google.oauth2 import service_account
# cred = service_account.Credentials.from_service_account_file(filepath_credentials)


# client = bigquery.Client(
#     project=PROJECT_ID,
#     credentials=cred,
#     _http=s
# )


# ##  3.
# client = bigquery.Client.from_service_account_json(
#     json_credentials_path=filepath_credentials,
#     _http=s
# )


# ## 4. Setting https_proxy environment variable
client = bigquery.Client(
    project=PROJECT_ID,
    credentials=cred,
    _http=s
)



sql = """
    SELECT name
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE state = `TX`
    LIMIT 100
"""

df = client.query(sql).to_dataframe()
