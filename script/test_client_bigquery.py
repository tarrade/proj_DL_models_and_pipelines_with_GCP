"""
running example with proxy.
authentification using skd credentials.


#option 1
! create `config_client.py` with proxy_https, filepath_ssl_cert and PROJECT_ID Strings:

proxy_https = "https://proxy-adress-incl-login-if-necessary"
filepath_ssl_cert = "C:/Users/path/to/sslfile"
PROJECT_ID = "name-gcp-project-12345"

#option 2
set all relevant variables as user environment variables
    1. search "env" in windows search bar (press windows button)
    2. select "Edit environment variables for your account"
    3. select "new" and add the PROXY_HTTPS, REQUESTS_CA_BUNDLE, PROJECT_ID, 
    maybe GOOGLE_APPLICATION_CREDENTDIALS if using a service account

"""
# pip install --upgrade google-cloud-bigquery
from google.cloud import bigquery

# # config_client contains some string variables defining key settings:
# from config_client import proxy_https
# from config_client import filepath_ssl_cert
# from config_client import PROJECT_ID

import os
# os.environ['PROJECT_ID'] = PROJECT_ID
# os.environ['https_proxy'] = proxy_https
# os.environ['REQUESTS_CA_BUNDLE'] = filepath_ssl_cert

# # Above environment variables can be set permanently 

PROJECT_ID = os.environ['PROJECT_ID']
print("# Current project in use: {}\n".format(PROJECT_ID))
client = bigquery.Client(project=PROJECT_ID)

sql = """
    SELECT *
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE state = 'TX'
    LIMIT 10
"""
df = client.query(sql).to_dataframe()
print(df)
