"""
Load natality dataset
- get two distinct datasets using hashs

Source:
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/2_sample.ipynb
"""

# pip install --upgrade google-cloud-bigquery
from google.cloud import bigquery as bq

# config_client contains some string variables defining key settings:
from config_client import proxy_https
from config_client import filepath_ssl_cert
from config_client import PROJECT_ID

import os
os.environ['PROJECT_ID'] = PROJECT_ID
os.environ['HTTPS_PROXY'] = proxy_https
os.environ['REQUESTS_CA_BUNDLE'] = filepath_ssl_cert


#To add:
#import client_utils as utils
#client = utils.get_bq_client(config="config_clients.yaml")

client = bq.Client()

query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""

# Call BigQuery but GROUP BY the hashmonth
# and see number of records for each group to enable us
# to get the correct train and evaluation percentages
df = client.query("SELECT hashmonth, COUNT(weight_pounds) AS num_babies FROM (" +
              query + ") GROUP BY hashmonth").to_dataframe()
print("There are {} unique hashmonths.".format(len(df)))
print(df.head())

# Added the RAND() so that we can now subsample from each of the hashmonths
# to get approximately the record counts we want
trainQuery = "SELECT * FROM (" + query + \
    ") WHERE MOD(hashmonth, 4) < 3 AND RAND() < 0.0005"
evalQuery = "SELECT * FROM (" + query + \
    ") WHERE MOD(hashmonth, 4) = 3 AND RAND() < 0.0005"
traindf = client.query(trainQuery).to_dataframe()
evaldf = client.query(evalQuery).to_dataframe()
print("There are {} examples in the train dataset and {} in the eval dataset".format(
    len(traindf), len(evaldf)))

# It is always crucial to clean raw data before using in ML, so we have a preprocessing step
import pandas as pd
def preprocess(df):
  # clean up data we don't want to train on
  # in other words, users will have to tell us the mother's age
  # otherwise, our ML service won't work.
  # these were chosen because they are such good predictors
  # and because these are easy enough to collect
  df = df[df.weight_pounds > 0]
  df = df[df.mother_age > 0]
  df = df[df.gestation_weeks > 0]
  df = df[df.plurality > 0]
  
  # modify plurality field to be a string
  twins_etc = dict(zip([1,2,3,4,5],
                   ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']))
  df['plurality'].replace(twins_etc, inplace=True)
  
  # now create extra rows to simulate lack of ultrasound
  nous = df.copy(deep=True)
  nous.loc[nous['plurality'] != 'Single(1)', 'plurality'] = 'Multiple(2+)'
  nous['is_male'] = 'Unknown'
  
  return pd.concat([df, nous])

print(traindf.head()) # Let's see a small sample of the training data now after our preprocessing
traindf = preprocess(traindf)
evaldf = preprocess(evaldf)
print(traindf.head())

traindf.to_csv('data/natality/train.csv', index=False, header=False)
evaldf.to_csv('data/natality/eval.csv', index=False, header=False)