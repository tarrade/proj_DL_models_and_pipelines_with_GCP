# pip installl --upgrade google-cloud-bigquery
client = bigquery.Client()
sql = """
    SELECT name
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE state = `TX`
    LIMIT 100
"""

df = client.query(sql).to_dataframe()