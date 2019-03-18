from config_client import proxy_https
from client_utils import no_ssl_verification
from google.cloud import storage
import os

os.environ['https_proxy'] = proxy_https

# If you don't specify credentials when constructing the client, the
# client library will look for credentials in the environment.
storage_client = storage.Client() # use current gcloud PROJECT_ID

# Make an authenticated API request
with no_ssl_verification():
    buckets = list(storage_client.list_buckets())
    print(buckets)